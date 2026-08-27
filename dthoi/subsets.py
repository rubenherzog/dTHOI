from __future__ import annotations

from itertools import combinations, islice
import math
from typing import Iterator

import torch


def canonicalize_subsets(subsets: torch.Tensor, n_variables: int | None = None) -> torch.Tensor:
    """Validate and sort a batch of fixed-order subsets."""

    subsets = torch.as_tensor(subsets)
    if subsets.ndim == 1:
        subsets = subsets.unsqueeze(0)
    if subsets.ndim != 2:
        raise ValueError("Subsets must have shape [B, K] or [K].")
    if subsets.shape[1] == 0:
        raise ValueError("Subset order must be at least one.")
    if subsets.dtype == torch.bool or subsets.dtype.is_floating_point:
        raise TypeError("Subset indices must use an integer dtype.")

    subsets = torch.sort(subsets.to(dtype=torch.long), dim=1).values
    if subsets.shape[1] > 1 and bool(torch.any(subsets[:, 1:] == subsets[:, :-1])):
        raise ValueError("A subset cannot contain duplicate variable indices.")
    if bool(torch.any(subsets < 0)):
        raise ValueError("Subset indices must be non-negative.")
    if n_variables is not None and bool(torch.any(subsets >= n_variables)):
        raise ValueError("Subset index exceeds the number of variables.")
    return subsets.contiguous()


def subset_to_mask(subset: torch.Tensor | list[int] | tuple[int, ...]) -> int:
    """Convert one subset to an arbitrary-width Python-integer bit mask."""

    mask = 0
    for idx in subset:
        mask |= 1 << int(idx)
    return mask


def masks_from_subsets(subsets: torch.Tensor) -> list[int]:
    subsets = canonicalize_subsets(subsets)
    return [subset_to_mask(row.tolist()) for row in subsets]


def mask_to_subset(mask: int) -> tuple[int, ...]:
    if mask < 0:
        raise ValueError("Mask must be non-negative.")
    out: list[int] = []
    i = 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return tuple(out)


def iter_subset_batches(
    n_variables: int,
    order: int,
    batch_size: int,
    *,
    device: torch.device | str | None = None,
) -> Iterator[torch.Tensor]:
    """Lazily generate fixed-order combinations directly in batches."""

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
    return math.comb(n_variables, order)
