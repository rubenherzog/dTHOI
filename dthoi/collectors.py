from __future__ import annotations

from typing import Any, Callable

import torch

from .measures.core import MeasureOutput

BatchCollector = Callable[[torch.Tensor, MeasureOutput, int, int], Any]
BatchAggregation = Callable[[list[Any]], Any]


def identity_collector(
    subsets: torch.Tensor,
    measures: MeasureOutput,
    order: int,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    tuple[torch.Tensor, tuple[torch.Tensor, ...]],
]:
    """Collect one completed batch on CPU.

    ``order`` and ``batch_index`` are accepted to preserve the common collector
    callback signature used by custom collectors. Local arrays are transferred
    to CPU together with the corresponding global batch so accelerator memory
    can be released before the next generated variable-set batch.
    """
    del order, batch_index
    subsets_cpu = subsets.detach().cpu()
    if isinstance(measures, tuple):
        values, local = measures
        return subsets_cpu, (
            values.detach().cpu(),
            tuple(dataset_values.detach().cpu() for dataset_values in local),
        )
    return subsets_cpu, measures.detach().cpu()


def identity_aggregation(items: list[Any]) -> list[Any]:
    """Return collected batch items without additional aggregation."""
    return items
