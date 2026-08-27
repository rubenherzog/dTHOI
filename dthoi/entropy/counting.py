from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CountMode = Literal["auto", "dense", "sparse"]

_INT64_BYTES = torch.empty((), dtype=torch.int64).element_size()
_FLOAT64_BYTES = torch.empty((), dtype=torch.float64).element_size()


@dataclass(frozen=True)
class SparseCounts:
    """Packed observed-state counts for a batch of variable sets.

    Parameters
    ----------
    values
        Positive state counts concatenated across all batch rows.
    row_ids
        Batch-row index corresponding to each value in ``values``.
    batch_size
        Number of variable sets represented.

    Notes
    -----
    The packed representation avoids Python loops over variable sets while
    allowing each set to contain a different number of observed states.
    """

    values: torch.Tensor
    row_ids: torch.Tensor
    batch_size: int


def encode_binary_states(data: torch.Tensor, subsets: torch.Tensor) -> torch.Tensor:
    """Encode binary subset configurations as compact integer states.

    Parameters
    ----------
    data
        Binary observations with shape ``[T, N]``.
    subsets
        Fixed-order variable subsets with shape ``[B, K]``.

    Returns
    -------
    torch.Tensor
        Exact ``torch.int64`` state codes with shape ``[B, T]``. Subset bits are
        packed contiguously, so codes lie in ``[0, 2**K)``.
    """
    if data.ndim != 2:
        raise ValueError("data must have shape [T, N].")
    if subsets.ndim != 2:
        raise ValueError("subsets must have shape [B, K].")

    batch_size, order = subsets.shape
    n_samples = data.shape[0]
    if order > 63:
        raise NotImplementedError(
            "The exact binary state encoder supports subset orders up to 63."
        )

    codes = torch.zeros((batch_size, n_samples), dtype=torch.int64, device=data.device)
    for bit in range(order):
        selected = data[:, subsets[:, bit]].transpose(0, 1).to(dtype=torch.int64)
        codes.bitwise_or_(selected << bit)
    return codes


def encode_binary_rows(data: torch.Tensor) -> torch.Tensor:
    """Encode complete binary observations into exact single-word states.

    The representation is available for datasets with at most 63 variables and
    is useful for repeatedly masking many sparse subset states.
    """
    if data.ndim != 2:
        raise ValueError("data must have shape [T, N].")

    n_samples, n_variables = data.shape
    if n_variables > 63:
        raise NotImplementedError("Single-word row encoding supports at most 63 variables.")

    codes = torch.zeros(n_samples, dtype=torch.int64, device=data.device)
    for bit in range(n_variables):
        codes.bitwise_or_(data[:, bit].to(dtype=torch.int64) << bit)
    return codes


def mask_binary_rows(row_codes: torch.Tensor, subset_masks: list[int]) -> torch.Tensor:
    """Apply canonical subset masks to pre-encoded binary observations.

    Returns a ``[B, T]`` tensor whose values preserve original variable bit
    positions. Contiguous state labels are unnecessary for observed-state
    counting.
    """
    if row_codes.ndim != 1:
        raise ValueError("row_codes must have shape [T].")

    masks = torch.tensor(subset_masks, dtype=torch.int64, device=row_codes.device)
    return row_codes.unsqueeze(0).bitwise_and(masks.unsqueeze(1))


def dense_counts_from_codes(codes: torch.Tensor, n_states: int) -> torch.Tensor:
    """Count every possible state for each encoded subset in one Torch batch.

    Parameters
    ----------
    codes
        Integer state codes with shape ``[B, T]`` and values in
        ``[0, n_states)``.
    n_states
        Total state-space cardinality used for each histogram.

    Returns
    -------
    torch.Tensor
        ``torch.int64`` counts with shape ``[B, n_states]``.
    """
    if codes.ndim != 2:
        raise ValueError("codes must have shape [B, T].")

    counts = torch.zeros(
        (codes.shape[0], n_states), dtype=torch.int64, device=codes.device
    )
    return counts.scatter_add_(1, codes, torch.ones_like(codes, dtype=torch.int64))


def sparse_counts_from_codes(
    codes: torch.Tensor,
    *,
    return_inverse: bool = False,
) -> SparseCounts | tuple[SparseCounts, torch.Tensor]:
    """Count observed states for every subset as one packed Torch batch.

    Parameters
    ----------
    codes
        Integer state codes with shape ``[B, T]``. Codes need not be contiguous
        across the state space.
    return_inverse
        If ``True``, also return a ``[B, T]`` tensor mapping every observation
        in original sample order to the corresponding position in packed
        ``values``. This supports local entropy calculation without state
        dictionaries or per-subset Python loops.

    Returns
    -------
    SparseCounts or tuple[SparseCounts, torch.Tensor]
        Packed positive counts and their batch-row indices. When requested, the
        second item is the packed-state index for every original observation.
    """
    if codes.ndim != 2:
        raise ValueError("codes must have shape [B, T].")

    batch_size, n_samples = codes.shape
    if n_samples == 0:
        empty = torch.empty(0, dtype=torch.int64, device=codes.device)
        packed = SparseCounts(empty, empty, batch_size)
        if return_inverse:
            return packed, torch.empty_like(codes, dtype=torch.int64)
        return packed

    sorted_codes, permutation = torch.sort(codes, dim=1)
    flattened = sorted_codes.reshape(-1)
    changes = torch.ones(flattened.numel(), dtype=torch.bool, device=codes.device)
    if flattened.numel() > 1:
        changes[1:] = flattened[1:] != flattened[:-1]
    if batch_size > 1:
        row_starts = torch.arange(1, batch_size, device=codes.device) * n_samples
        changes[row_starts] = True

    starts = torch.nonzero(changes, as_tuple=False).flatten()
    final = torch.tensor([batch_size * n_samples], dtype=starts.dtype, device=codes.device)
    ends = torch.cat((starts[1:], final))
    values = (ends - starts).to(dtype=torch.int64)
    row_ids = torch.div(starts, n_samples, rounding_mode="floor").to(dtype=torch.int64)
    packed = SparseCounts(values=values, row_ids=row_ids, batch_size=batch_size)

    if not return_inverse:
        return packed

    packed_index_sorted = (
        torch.cumsum(changes.to(dtype=torch.int64), dim=0) - 1
    ).reshape(batch_size, n_samples)
    inverse = torch.empty_like(packed_index_sorted)
    inverse.scatter_(1, permutation, packed_index_sorted)
    return packed, inverse


def choose_count_mode(
    *,
    order: int,
    n_subsets: int,
    n_samples: int,
    requested: CountMode,
    dense_state_limit: int,
    dense_memory_limit_bytes: int,
) -> Literal["dense", "sparse"]:
    """Select dense or observed-state counting for one entropy workload.

    ``auto`` uses state-space size, sample count, and a conservative estimate of
    the dense count plus float64 entropy working memory. Explicit ``dense`` or
    ``sparse`` requests are returned unchanged.
    """
    if requested in {"dense", "sparse"}:
        return requested
    if requested != "auto":
        raise ValueError("count_mode must be 'auto', 'dense', or 'sparse'.")

    n_states = 1 << order
    bytes_per_state = _INT64_BYTES + 3 * _FLOAT64_BYTES
    dense_working_bytes = n_subsets * n_states * bytes_per_state

    if n_states > dense_state_limit or dense_working_bytes > dense_memory_limit_bytes:
        return "sparse"

    return "dense" if n_states <= max(4 * n_samples, 256) else "sparse"
