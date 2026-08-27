from __future__ import annotations

from typing import Any

import torch

from .collectors import BatchAggregation, BatchCollector, identity_aggregation, identity_collector
from .data import PreparedDiscreteData, prepare_discrete_data
from .entropy.base import CountingEntropyProvider
from .entropy.cache import EntropyCache
from .entropy.counting import CountMode
from .entropy.estimators import CountEntropyEstimator
from .measures.core import measures_from_provider
from .subsets import iter_subset_batches


def multi_order_measures(
    X: PreparedDiscreteData | torch.Tensor | list[torch.Tensor],
    *,
    min_order: int = 3,
    max_order: int | None = None,
    estimator: str | CountEntropyEstimator = "plugin",
    count_mode: CountMode = "auto",
    batch_size: int = 4096,
    device: torch.device | str | None = None,
    cache: EntropyCache | None = None,
    batch_collector: BatchCollector | None = None,
    aggregation: BatchAggregation | None = None,
) -> Any:
    """Exhaustively evaluate a range of subset orders using one entropy cache."""

    prepared = X if isinstance(X, PreparedDiscreteData) else prepare_discrete_data(X, device=device)
    if device is not None and isinstance(X, PreparedDiscreteData):
        target = torch.device(device)
        if any(ds.device != target for ds in X.datasets):
            prepared = PreparedDiscreteData(tuple(ds.to(target) for ds in X.datasets), X.n_variables)

    N = prepared.n_variables
    max_order = N if max_order is None else max_order
    if not 2 <= min_order <= max_order <= N:
        raise ValueError("Require 2 <= min_order <= max_order <= N.")

    entropy_cache = cache if cache is not None else EntropyCache()
    provider = CountingEntropyProvider(
        prepared,
        estimator=estimator,
        count_mode=count_mode,
        cache=entropy_cache,
    )

    collector = identity_collector if batch_collector is None else batch_collector
    aggregate = identity_aggregation if aggregation is None else aggregation

    items: list[Any] = []
    for order in range(min_order, max_order + 1):
        for batch_index, subsets in enumerate(
            iter_subset_batches(N, order, batch_size, device=prepared.device)
        ):
            measures = measures_from_provider(provider, subsets)
            items.append(collector(subsets, measures, order, batch_index))

    return aggregate(items)
