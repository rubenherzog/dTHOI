from __future__ import annotations

from typing import Any, Callable

import torch

BatchCollector = Callable[[torch.Tensor, torch.Tensor, int, int], Any]
BatchAggregation = Callable[[list[Any]], Any]


def identity_collector(
    subsets: torch.Tensor,
    measures: torch.Tensor,
    order: int,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Default memory-safe collector: move completed batches to CPU."""

    return subsets.detach().cpu(), measures.detach().cpu()


def identity_aggregation(items: list[Any]) -> list[Any]:
    return items
