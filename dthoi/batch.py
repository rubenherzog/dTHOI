from __future__ import annotations

from typing import Any

import torch

from .collectors import BatchAggregation, BatchCollector, identity_aggregation, identity_collector
from .data import DiscreteInput, prepare_discrete_data
from .entropy.cache import EntropyCache
from .entropy.counting import CountMode
from .entropy.estimators import CountEntropyEstimator
from .measures.core import DEFAULT_MAX_LOCAL_VALUES_PER_BATCH, local_sets_per_batch
from .measures.dispatch import make_measure_provider
from .subsets import iter_subset_batches


def multi_order_measures(
    X: DiscreteInput,
    *,
    min_order: int = 3,
    max_order: int | None = None,
    estimator: str | CountEntropyEstimator = "empirical",
    count_mode: CountMode = "auto",
    batch_size: int = 4096,
    device: torch.device | str | None = None,
    cache: EntropyCache | None = None,
    batch_collector: BatchCollector | None = None,
    aggregation: BatchAggregation | None = None,
    local_values: bool = False,
    max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
) -> Any:
    """Evaluate all subsets across a contiguous range of interaction orders.

    Subsets are generated lazily and evaluated through the same estimator
    dispatcher used by :func:`dthoi.information_measures`. Entropy-composition
    estimators reuse one shared entropy provider and cache across all orders;
    complete-joint estimators reuse one prepared data representation and apply
    estimator-specific bounded working batches.

    Parameters
    ----------
    X
        Discrete observations accepted by :func:`dthoi.prepare_data`.
    min_order, max_order
        Inclusive interaction-order range. ``max_order=None`` uses all variables.
    estimator
        Information-measure estimator name or an entropy-estimator instance.
    count_mode
        ``"dense"``, ``"sparse"``, or ``"auto"`` for entropy-based estimators.
    batch_size
        Maximum number of generated variable sets per returned batch.
    device
        Destination device for observations and subset batches.
    cache
        Optional entropy cache. It is used only by entropy-composition
        estimators because coherent marginal entropies depend on their parent set.
    batch_collector
        Callback receiving ``(subsets, measures, order, batch_index)``.
    aggregation
        Callback applied once to the collected batch outputs.
    local_values
        Whether to calculate sample-resolved values. Only estimators with an
        explicit local-value definition support this option.
    max_local_values_per_batch
        Target upper bound for ``variable sets × total samples`` in local work.

    Returns
    -------
    Any
        Aggregated collector output. By default, a list of CPU tuples.
    """
    prepared = prepare_discrete_data(X, device=device)
    n_variables = prepared.n_variables
    max_order = n_variables if max_order is None else max_order
    if not 2 <= min_order <= max_order <= n_variables:
        raise ValueError("Require 2 <= min_order <= max_order <= N.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    provider = make_measure_provider(
        prepared,
        estimator=estimator,
        count_mode=count_mode,
        cache=cache,
    )
    if local_values and not provider.supports_local_values:
        raise NotImplementedError(
            f"Estimator {provider.name!r} does not define sample-resolved local values."
        )

    collector = identity_collector if batch_collector is None else batch_collector
    aggregate = identity_aggregation if aggregation is None else aggregation

    items: list[Any] = []
    for order in range(min_order, max_order + 1):
        effective_batch_size = batch_size
        preferred = provider.preferred_batch_size(order)
        if preferred is not None:
            effective_batch_size = min(effective_batch_size, preferred)
        if local_values:
            effective_batch_size = min(
                effective_batch_size,
                local_sets_per_batch(prepared, max_local_values_per_batch),
            )

        for batch_index, subsets in enumerate(
            iter_subset_batches(
                n_variables,
                order,
                effective_batch_size,
                device=prepared.device,
            )
        ):
            measures = provider.measures(
                subsets,
                local_values=local_values,
                max_local_values_per_batch=max_local_values_per_batch,
            )
            items.append(collector(subsets, measures, order, batch_index))

    return aggregate(items)
