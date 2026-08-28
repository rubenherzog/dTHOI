from __future__ import annotations

from itertools import combinations, islice
import math
from typing import Iterator

import torch


def canonicalize_subsets(subsets: torch.Tensor, n_variables: int | None = None) -> torch.Tensor:
    """Validate and sort a fixed-order batch of variable subsets.

    Parameters
    ----------
    subsets
        Integer tensor with shape ``[B, K]`` or one subset with shape ``[K]``.
    n_variables
        Optional upper bound for variable indices.

    Returns
    -------
    torch.Tensor
        Contiguous ``torch.long`` tensor with shape ``[B, K]`` and each row
        sorted in ascending order.
    """
    subsets = torch.as_tensor(subsets)
    if subsets.ndim == 1:
        subsets = subsets.unsqueeze(0)
    if subsets.ndim != 2:
        raise ValueError("Subsets must have shape [B, K] or [K].")
    if subsets.shape[1] == 0:
        raise ValueError("Subset order must be at least one.")
    if subsets.dtype == torch.bool or subsets.dtype.is_floating_point:
        raise TypeError("Subset indices must use an integer dtype.")

    subsets = subsets.to(dtype=torch.long)
    if subsets.shape[1] > 1 and bool(torch.any(subsets[:, 1:] < subsets[:, :-1])):
        subsets = torch.sort(subsets, dim=1).values
    if subsets.shape[1] > 1 and bool(torch.any(subsets[:, 1:] == subsets[:, :-1])):
        raise ValueError("A subset cannot contain duplicate variable indices.")
    if bool(torch.any(subsets < 0)):
        raise ValueError("Subset indices must be non-negative.")
    if n_variables is not None and bool(torch.any(subsets >= n_variables)):
        raise ValueError("Subset index exceeds the number of variables.")
    return subsets.contiguous()


def subset_to_mask(subset: torch.Tensor | list[int] | tuple[int, ...]) -> int:
    """Convert one subset to an arbitrary-width Python integer bit mask."""
    indices = subset.detach().cpu().tolist() if isinstance(subset, torch.Tensor) else subset
    mask = 0
    for idx in indices:
        mask |= 1 << int(idx)
    return mask


def _masks_from_canonical_subsets(subsets: torch.Tensor) -> list[int]:
    """Convert canonical subsets to cache keys with one device transfer at most."""
    if subsets.numel() == 0:
        return []

    if int(subsets.max()) < 63:
        bits = torch.ones_like(subsets, dtype=torch.int64) << subsets
        return bits.sum(dim=1).detach().cpu().tolist()

    rows = subsets.detach().cpu().tolist()
    return [subset_to_mask(row) for row in rows]


def masks_from_subsets(subsets: torch.Tensor) -> list[int]:
    """Convert a subset batch to order-independent Python integer bit masks."""
    return _masks_from_canonical_subsets(canonicalize_subsets(subsets))


def deduplicate_subsets(
    subsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Deduplicate fixed-order variable sets while preserving first occurrence.

    Parameters
    ----------
    subsets
        Integer variable sets with shape ``[B, K]`` or one set ``[K]``.

    Returns
    -------
    tuple
        ``(unique_subsets, inverse, unique_masks)``. ``inverse`` maps each input
        row to its corresponding row in ``unique_subsets``. Python-integer masks
        are returned with the unique rows so callers can reuse the same exact
        cache/counting keys without recomputing them.

    Notes
    -----
    Python integers are intentionally used for the keys because the full dTHOI
    data model allows hundreds of variables even when a selected interaction
    order exceeds a single machine word.
    """
    canonical = canonicalize_subsets(subsets)
    if canonical.shape[0] == 0:
        inverse = torch.empty(0, dtype=torch.long, device=canonical.device)
        return canonical, inverse, []

    masks = _masks_from_canonical_subsets(canonical)
    unique_rows: list[torch.Tensor] = []
    unique_masks: list[int] = []
    index_by_mask: dict[int, int] = {}
    inverse_positions: list[int] = []

    for mask, subset in zip(masks, canonical, strict=True):
        index = index_by_mask.get(mask)
        if index is None:
            index = len(unique_rows)
            index_by_mask[mask] = index
            unique_rows.append(subset)
            unique_masks.append(mask)
        inverse_positions.append(index)

    return (
        torch.stack(unique_rows),
        torch.tensor(inverse_positions, dtype=torch.long, device=canonical.device),
        unique_masks,
    )


def mask_to_subset(mask: int) -> tuple[int, ...]:
    """Convert a non-negative Python integer bit mask back to variable indices."""
    if mask < 0:
        raise ValueError("Mask must be non-negative.")
    out: list[int] = []
    index = 0
    while mask:
        if mask & 1:
            out.append(index)
        mask >>= 1
        index += 1
    return tuple(out)


def leave_one_out_subsets(subsets: torch.Tensor) -> torch.Tensor:
    """Generate every one-variable deletion from a canonical subset batch.

    Parameters
    ----------
    subsets
        Canonical subset tensor with shape ``[B, K]`` and ``K >= 2``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[B * K, K - 1]`` ordered first by input subset and
        then by the removed variable position.
    """
    if subsets.ndim != 2 or subsets.shape[1] < 2:
        raise ValueError("subsets must have shape [B, K] with K >= 2.")

    batch_size, order = subsets.shape
    indices = torch.arange(order, device=subsets.device)
    keep = indices.unsqueeze(0) != indices.unsqueeze(1)
    loo_indices = indices.expand(order, order)[keep].reshape(order, order - 1)
    return subsets[:, loo_indices].reshape(batch_size * order, order - 1)


def iter_subset_batches(
    n_variables: int,
    order: int,
    batch_size: int,
    *,
    device: torch.device | str | None = None,
) -> Iterator[torch.Tensor]:
    """Lazily yield fixed-order combinations as Torch batches.

    The global combination space is never materialized. Python's exact
    combination iterator is chunked and each chunk is converted once to a
    ``torch.long`` tensor.
    """
    if not 1 <= order <= n_variables:
        raise ValueError("order must satisfy 1 <= order <= n_variables.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    iterator = combinations(range(n_variables), order)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            return
        yield torch.tensor(chunk, dtype=torch.long, device=device)


def n_subsets(n_variables: int, order: int) -> int:
    """Return the exact number of unordered subsets ``C(N, K)``."""
    return math.comb(n_variables, order)
