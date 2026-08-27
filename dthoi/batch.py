from __future__ import annotations

from typing import Any

import torch

from .collectors import BatchAggregation, BatchCollector, identity_aggregation, identity_collector
from .data import DiscreteInput, prepare_discrete_data
from .entropy.base import CountingEntropyProvider
from .entropy.cache import EntropyCache
from .entropy.counting import CountMode
from .entropy.estimators import CountEntropyEstimator
from .measures.core import measures_from_provider
from .subsets import iter_subset_batches


def multi_order_measures(
    X: DiscreteInput,
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
    """Evaluate all subsets across a contiguous range of orders.

    Subsets are generated lazily in batches and evaluated through one shared
    entropy provider and cache. Completed batches are moved to CPU by the
    default collector.

    This is an internal execution function. Scientific users should normally
    call :func:`dthoi.analyze_orders`.

    Parameters
    ----------
    X
        Discrete observations accepted by :func:`dthoi.prepare_data`.
    min_order, max_order
        Inclusive subset-order range. ``max_order=None`` uses all variables.
    estimator
        Entropy estimator name or estimator instance.
    count_mode
        ``"dense"``, ``"sparse"``, or ``"auto"`` counting strategy.
    batch_size
        Maximum number of subsets generated per batch.
    device
        Destination device for the observations and subset batches.
    cache
        Optional entropy cache shared across every order and batch.
    batch_collector
        Callback receiving ``(subsets, measures, order, batch_index)``.
    aggregation
        Callback applied once to the list of collected batch outputs.

    Returns
    -------
    Any
        Aggregated collector output. By default, a list of CPU
        ``(subsets, measures)`` tuples.
    """
    prepared = prepare_discrete_data(X, device=device)

    n_variables = prepared.n_variables
    max_order = n_variables if max_order is None else max_order
    if not 2 <= min_order <= max_order <= n_variables:
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
            iter_subset_batches(n_variables, order, batch_size, device=prepared.device)
        ):
            measures = measures_from_provider(provider, subsets)
            items.append(collector(subsets, measures, order, batch_index))

    return aggregate(items)
