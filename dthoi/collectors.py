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
    """Collect one completed batch on CPU.

    ``order`` and ``batch_index`` are accepted to preserve the common collector
    callback signature used by custom collectors.
    """
    return subsets.detach().cpu(), measures.detach().cpu()


def identity_aggregation(items: list[Any]) -> list[Any]:
    """Return collected batch items without additional aggregation."""
    return items
