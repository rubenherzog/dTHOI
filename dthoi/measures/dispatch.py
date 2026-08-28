"""Internal estimator dispatch for higher-order information measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..data import DiscreteData, DiscreteInput, prepare_discrete_data
from ..entropy.base import CountingEntropyProvider
from ..entropy.cache import EntropyCache
from ..entropy.counting import CountMode
from ..entropy.estimators import CountEntropyEstimator, resolve_estimator
from ..subsets import canonicalize_subsets
from .coherent import COHERENT_ESTIMATORS, coherent_measures, normalize_coherent_estimator
from .core import (
    DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
    local_sets_per_batch,
    measures_from_provider,
)
from .definitions import LocalMeasureValues, MeasureOutput

EstimatorFamily = Literal["entropy", "coherent"]
MeasureEstimatorInput = str | CountEntropyEstimator

_COHERENT_WORK_VALUE_TARGET = 8_000_000
_COHERENT_GRID_POINTS = 72


@dataclass(frozen=True)
class ResolvedMeasureEstimator:
    """Resolved estimator family and implementation used by measure providers."""

    name: str
    family: EstimatorFamily
    entropy_estimator: CountEntropyEstimator | None
    supports_local_values: bool


def resolve_measure_estimator(estimator: MeasureEstimatorInput) -> ResolvedMeasureEstimator:
    """Resolve one estimator name across entropy-based and coherent families."""
    if isinstance(estimator, CountEntropyEstimator):
        return ResolvedMeasureEstimator(
            name=estimator.name,
            family="entropy",
            entropy_estimator=estimator,
            supports_local_values=estimator.supports_local_values,
        )
    if not isinstance(estimator, str):
        raise TypeError("estimator must be a string or CountEntropyEstimator instance.")

    key = estimator.lower().replace("-", "_")
    if key in COHERENT_ESTIMATORS:
        name = normalize_coherent_estimator(key)
        return ResolvedMeasureEstimator(
            name=name,
            family="coherent",
            entropy_estimator=None,
            supports_local_values=False,
        )

    entropy = resolve_estimator(estimator)
    return ResolvedMeasureEstimator(
        name=entropy.name,
        family="entropy",
        entropy_estimator=entropy,
        supports_local_values=entropy.supports_local_values,
    )


def resolve_public_estimator_argument(
    *,
    estimator: str | None,
    entropy_estimator: str | None,
) -> str:
    """Resolve the public estimator argument while preserving the legacy alias."""
    if estimator is not None and entropy_estimator is not None:
        raise ValueError("Specify only one of estimator= or entropy_estimator=.")
    if estimator is not None:
        return estimator
    if entropy_estimator is not None:
        return entropy_estimator
    return "empirical"


class EntropyMeasureProvider:
    """Measure provider that composes shared subset entropy estimates."""

    def __init__(
        self,
        X: DiscreteInput,
        *,
        estimator: CountEntropyEstimator,
        count_mode: CountMode,
        cache: EntropyCache | None,
        device: torch.device | str | None,
    ) -> None:
        """Prepare one reusable entropy provider for repeated measure batches."""
        self.entropy_provider = CountingEntropyProvider(
            X,
            estimator=estimator,
            count_mode=count_mode,
            cache=cache,
            device=device,
        )
        self.data = self.entropy_provider.data
        self.name = estimator.name
        self.supports_local_values = estimator.supports_local_values

    def preferred_batch_size(self, order: int) -> int | None:
        """Return no additional global batch constraint for entropy composition."""
        del order
        return None

    def measures(
        self,
        subsets: torch.Tensor,
        *,
        local_values: bool = False,
        max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
    ) -> MeasureOutput:
        """Calculate information measures from the shared entropy provider."""
        return measures_from_provider(
            self.entropy_provider,
            subsets,
            local_values=local_values,
            max_local_values_per_batch=max_local_values_per_batch,
        )


class CoherentMeasureProvider:
    """Measure provider for complete-joint coherent Bayesian estimators."""

    def __init__(
        self,
        X: DiscreteInput,
        *,
        estimator: str,
        device: torch.device | str | None,
    ) -> None:
        """Prepare data once and retain the coherent estimator name."""
        self.data: DiscreteData = prepare_discrete_data(X, device=device)
        self.estimator = normalize_coherent_estimator(estimator)
        self.name = self.estimator
        self.supports_local_values = False

    def preferred_batch_size(self, order: int) -> int | None:
        """Return a conservative batch bound for posterior-grid working memory."""
        if order < 2:
            return None
        grid_points = 1 if self.estimator == "dirichlet_a1" else _COHERENT_GRID_POINTS
        work_factor = max(order, grid_points)
        max_samples = max(self.data.sample_counts)
        return max(1, _COHERENT_WORK_VALUE_TARGET // (max_samples * work_factor))

    def measures(
        self,
        subsets: torch.Tensor,
        *,
        local_values: bool = False,
        max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
    ) -> torch.Tensor:
        """Calculate coherent posterior-mean information measures."""
        del max_local_values_per_batch
        if local_values:
            raise NotImplementedError(
                f"Estimator {self.estimator!r} does not define sample-resolved local values."
            )
        return coherent_measures(
            self.data,
            subsets,
            estimator=self.estimator,
        )


MeasureProvider = EntropyMeasureProvider | CoherentMeasureProvider


def make_measure_provider(
    X: DiscreteInput,
    *,
    estimator: MeasureEstimatorInput = "empirical",
    count_mode: CountMode = "auto",
    cache: EntropyCache | None = None,
    device: torch.device | str | None = None,
) -> MeasureProvider:
    """Create one reusable provider for the selected information estimator."""
    resolved = resolve_measure_estimator(estimator)
    if resolved.family == "coherent":
        return CoherentMeasureProvider(X, estimator=resolved.name, device=device)

    assert resolved.entropy_estimator is not None
    return EntropyMeasureProvider(
        X,
        estimator=resolved.entropy_estimator,
        count_mode=count_mode,
        cache=cache,
        device=device,
    )


def measures_for_sets(
    X: DiscreteInput,
    subsets: torch.Tensor,
    *,
    estimator: MeasureEstimatorInput = "empirical",
    count_mode: CountMode = "auto",
    cache: EntropyCache | None = None,
    device: torch.device | str | None = None,
    local_values: bool = False,
    max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
) -> MeasureOutput:
    """Calculate higher-order information measures for explicit variable sets."""
    provider = make_measure_provider(
        X,
        estimator=estimator,
        count_mode=count_mode,
        cache=cache,
        device=device,
    )
    subsets = canonicalize_subsets(subsets, provider.data.n_variables).to(provider.data.device)

    if local_values and not provider.supports_local_values:
        raise NotImplementedError(
            f"Estimator {provider.name!r} does not define sample-resolved local values."
        )

    requested_batch = subsets.shape[0]
    preferred = provider.preferred_batch_size(subsets.shape[1])
    if preferred is not None:
        requested_batch = min(requested_batch, preferred)
    if local_values:
        requested_batch = min(
            requested_batch,
            local_sets_per_batch(provider.data, max_local_values_per_batch),
        )
    requested_batch = max(1, requested_batch)

    if requested_batch >= subsets.shape[0]:
        return provider.measures(
            subsets,
            local_values=local_values,
            max_local_values_per_batch=max_local_values_per_batch,
        )

    values_out = torch.empty(
        (subsets.shape[0], provider.data.n_datasets, 4),
        dtype=torch.float64,
        device=provider.data.device,
    )
    local_out: LocalMeasureValues | None = None
    if local_values:
        local_out = tuple(
            torch.empty(
                (subsets.shape[0], dataset.shape[0], 4),
                dtype=torch.float64,
                device=provider.data.device,
            )
            for dataset in provider.data.datasets
        )

    for start in range(0, subsets.shape[0], requested_batch):
        stop = min(start + requested_batch, subsets.shape[0])
        output = provider.measures(
            subsets[start:stop],
            local_values=local_values,
            max_local_values_per_batch=max_local_values_per_batch,
        )
        if isinstance(output, tuple):
            values, local = output
            values_out[start:stop] = values
            assert local_out is not None
            for dataset_index, dataset_values in enumerate(local):
                local_out[dataset_index][start:stop] = dataset_values
        else:
            values_out[start:stop] = output

    if local_out is not None:
        return values_out, local_out
    return values_out
