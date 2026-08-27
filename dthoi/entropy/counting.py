from __future__ import annotations

from typing import Literal

import torch

CountMode = Literal["auto", "dense", "sparse"]

_INT64_BYTES = torch.empty((), dtype=torch.int64).element_size()
_FLOAT64_BYTES = torch.empty((), dtype=torch.float64).element_size()


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
    """Count every possible state for each encoded subset.

    Parameters
    ----------
    codes
        Integer state codes with shape ``[B, T]``.
    n_states
        Total state-space cardinality used as ``minlength`` for each histogram.

    Returns
    -------
    torch.Tensor
        ``torch.int64`` counts with shape ``[B, n_states]``.
    """
    batch_size = codes.shape[0]
    counts = torch.zeros((batch_size, n_states), dtype=torch.int64, device=codes.device)
    for batch_index in range(batch_size):
        counts[batch_index] = torch.bincount(codes[batch_index], minlength=n_states)
    return counts


def sparse_counts_from_codes(codes: torch.Tensor) -> list[torch.Tensor]:
    """Count only observed states for each encoded subset.

    Parameters
    ----------
    codes
        Integer state codes with shape ``[B, T]``.

    Returns
    -------
    list[torch.Tensor]
        One ``torch.int64`` count vector per subset. Vector lengths equal the
        number of observed states and may differ between subsets.
    """
    values: list[torch.Tensor] = []
    for row in codes:
        sorted_codes = torch.sort(row).values
        _, counts = torch.unique_consecutive(sorted_codes, return_counts=True)
        values.append(counts)
    return values


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
