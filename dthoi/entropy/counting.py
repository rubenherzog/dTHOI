from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CountMode = Literal["auto", "dense", "sparse"]


@dataclass(frozen=True)
class DenseCounts:
    values: torch.Tensor  # [B, Q]


@dataclass(frozen=True)
class SparseCounts:
    values: list[torch.Tensor]  # one vector of observed-state counts per subset


def encode_binary_states(data: torch.Tensor, subsets: torch.Tensor) -> torch.Tensor:
    """Encode binary configurations without materializing ``[B, T, K]``.

    Returns
    -------
    torch.Tensor
        Integer state codes of shape ``[B, T]``.
    """

    if data.ndim != 2:
        raise ValueError("data must have shape [T, N].")
    if subsets.ndim != 2:
        raise ValueError("subsets must have shape [B, K].")

    B, K = subsets.shape
    T = data.shape[0]
    if K > 63:
        raise NotImplementedError(
            "The initial exact binary state encoder supports subset orders up to 63. "
            "Multi-word state codes are a later backend extension."
        )

    codes = torch.zeros((B, T), dtype=torch.int64, device=data.device)
    for bit in range(K):
        selected = data[:, subsets[:, bit]].transpose(0, 1).to(dtype=torch.int64)
        codes.bitwise_or_(selected << bit)
    return codes


def dense_counts_from_codes(codes: torch.Tensor, n_states: int) -> DenseCounts:
    B = codes.shape[0]
    counts = torch.zeros((B, n_states), dtype=torch.int64, device=codes.device)
    for b in range(B):
        counts[b] = torch.bincount(codes[b], minlength=n_states)
    return DenseCounts(counts)


def sparse_counts_from_codes(codes: torch.Tensor) -> SparseCounts:
    values: list[torch.Tensor] = []
    for row in codes:
        sorted_codes = torch.sort(row).values
        _, counts = torch.unique_consecutive(sorted_codes, return_counts=True)
        values.append(counts)
    return SparseCounts(values)


def choose_count_mode(
    *,
    order: int,
    n_subsets: int,
    n_samples: int,
    requested: CountMode,
    dense_state_limit: int,
    dense_memory_limit_bytes: int,
) -> Literal["dense", "sparse"]:
    if requested in {"dense", "sparse"}:
        return requested
    if requested != "auto":
        raise ValueError("count_mode must be 'auto', 'dense', or 'sparse'.")

    n_states = 1 << order
    dense_bytes = n_subsets * n_states * torch.tensor([], dtype=torch.int64).element_size()

    if n_states > dense_state_limit or dense_bytes > dense_memory_limit_bytes:
        return "sparse"

    # Dense counting is attractive when the potential state space is not much
    # larger than the number of observations actually available.
    return "dense" if n_states <= max(4 * n_samples, 256) else "sparse"
