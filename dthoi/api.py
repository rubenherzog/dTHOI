"""User-facing API for dTHOI.

The public API is intentionally phrased in scientific terms. Implementation
concepts such as counting strategies, entropy caches, and execution providers
remain internal so routine analyses can be expressed with a small set of
functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .batch import multi_order_measures
from .data import DiscreteData, prepare_discrete_data
from .entropy.base import CountingEntropyProvider
from .measures.core import (
    DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
    MeasureOutput,
    nplets_measures,
)


@dataclass(frozen=True)
class LocalInformationMeasures:
    """Sample-resolved higher-order information values.

    Parameters
    ----------
    values
        One tensor per dataset. Each tensor has shape
        ``[variable_sets, samples_d, 4]`` and the final dimension contains, in
        order, local total correlation, local dual total correlation, local
        O-information, and local S-information. Values are measured in bits.

    Notes
    -----
    Datasets remain separate so unequal sample counts are represented without
    padding or reshaping observations. The named properties are views of the
    stored tensors and do not duplicate the local arrays.
    """

    values: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        """Validate dataset-local tensor shapes."""
        if not self.values:
            raise ValueError("Local information must contain at least one dataset.")
        for dataset_values in self.values:
            if dataset_values.ndim != 3 or dataset_values.shape[-1] != 4:
                raise ValueError(
                    "Each local tensor must have shape [variable_sets, samples, 4]."
                )

    @property
    def total_correlation(self) -> tuple[torch.Tensor, ...]:
        """Local total correlation for each dataset, in bits."""
        return tuple(values[..., 0] for values in self.values)

    @property
    def dual_total_correlation(self) -> tuple[torch.Tensor, ...]:
        """Local dual total correlation for each dataset, in bits."""
        return tuple(values[..., 1] for values in self.values)

    @property
    def o_information(self) -> tuple[torch.Tensor, ...]:
        """Local O-information for each dataset, in bits."""
        return tuple(values[..., 2] for values in self.values)

    @property
    def s_information(self) -> tuple[torch.Tensor, ...]:
        """Local S-information for each dataset, in bits."""
        return tuple(values[..., 3] for values in self.values)

    def as_dict(self) -> dict[str, tuple[torch.Tensor, ...]]:
        """Return the four local measures with descriptive names."""
        return {
            "total_correlation": self.total_correlation,
            "dual_total_correlation": self.dual_total_correlation,
            "o_information": self.o_information,
            "s_information": self.s_information,
        }


@dataclass(frozen=True)
class InformationMeasures:
    """Higher-order information measures for one or more variable sets.

    Parameters
    ----------
    values
        Tensor whose final dimension contains, in order, total correlation,
        dual total correlation, O-information, and S-information. For the
        standard dTHOI API the shape is ``[variable_sets, datasets, 4]``.
    local
        Optional sample-resolved values requested with ``local_values=True``.
        The default is ``None`` so global analyses do not allocate arrays that
        scale with sample count.

    Notes
    -----
    The named properties are views of ``values`` and therefore do not duplicate
    the underlying data.
    """

    values: torch.Tensor
    local: LocalInformationMeasures | None = None

    def __post_init__(self) -> None:
        """Validate global and optional local measure dimensions."""
        if self.values.ndim < 1 or self.values.shape[-1] != 4:
            raise ValueError("The final dimension must contain the four dTHOI measures.")
        if self.local is not None:
            if self.values.ndim < 3:
                raise ValueError("Local information requires standard [sets, datasets, 4] values.")
            if len(self.local.values) != self.values.shape[-2]:
                raise ValueError("Local information must contain one tensor per dataset.")
            for dataset_values in self.local.values:
                if dataset_values.shape[0] != self.values.shape[0]:
                    raise ValueError(
                        "Global and local information must contain the same variable sets."
                    )

    @property
    def total_correlation(self) -> torch.Tensor:
        """Total correlation for each variable set and dataset."""
        return self.values[..., 0]

    @property
    def dual_total_correlation(self) -> torch.Tensor:
        """Dual total correlation for each variable set and dataset."""
        return self.values[..., 1]

    @property
    def o_information(self) -> torch.Tensor:
        """O-information for each variable set and dataset."""
        return self.values[..., 2]

    @property
    def s_information(self) -> torch.Tensor:
        """S-information for each variable set and dataset."""
        return self.values[..., 3]

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the four measures as a dictionary with descriptive names."""
        return {
            "total_correlation": self.total_correlation,
            "dual_total_correlation": self.dual_total_correlation,
            "o_information": self.o_information,
            "s_information": self.s_information,
        }


@dataclass(frozen=True)
class InteractionResults:
    """Results for a collection of variable sets of the same order.

    Parameters
    ----------
    order
        Number of variables in each set.
    variable_sets
        Integer tensor identifying the variables included in each set.
    information
        Higher-order information measures for those variable sets. Optional
        sample-resolved values are available through ``information.local``.
    """

    order: int
    variable_sets: torch.Tensor
    information: InformationMeasures


def prepare_data(
    data: Any,
    *,
    device: torch.device | str | None = None,
) -> DiscreteData:
    """Prepare discrete observations for repeated dTHOI analyses.

    Parameters
    ----------
    data
        Binary observations arranged as ``[samples, variables]``. Multiple
        datasets may be provided either as ``[datasets, samples, variables]``
        when sample counts are equal, or as a sequence of two-dimensional
        arrays/tensors when sample counts differ.
    device
        Optional Torch device on which calculations should be performed.

    Returns
    -------
    DiscreteData
        Canonical data representation that can be reused across analyses.
    """
    return prepare_discrete_data(data, device=device)


def estimate_entropy(
    data: Any,
    variable_sets: Any,
    *,
    entropy_estimator: str = "empirical",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Estimate Shannon entropy for selected sets of variables.

    Parameters
    ----------
    data
        Binary observations or a prepared :class:`DiscreteData` object.
    variable_sets
        Variable indices with shape ``[sets, order]`` or one set with shape
        ``[order]``. Variables may be supplied in any order.
    entropy_estimator
        Entropy estimator. Supported choices are ``"empirical"``,
        ``"miller_madow"``, ``"schurmann"``, ``"shrinkage"``,
        ``"chao_shen"``, ``"pitman_yor"``, and ``"ansb"``. Pitman-Yor uses
        the fixed convention ``d=1/2, alpha=0``. ANSB is intended for severe
        undersampling and warns when ``N / 2**order > 0.1``.
    device
        Optional Torch device on which calculations should be performed.

    Returns
    -------
    torch.Tensor
        Entropy in bits with shape ``[variable_sets, datasets]``.

    Notes
    -----
    All estimators reuse the same exact state-counting paths. Shrinkage uses the
    complete nominal binary support ``2**order``; if structural zeros are known
    scientifically, this uniform-target assumption should be considered.
    Chao-Shen returns ``NaN`` when every observation is a singleton. ANSB is
    undefined without repeated observations and returns ``NaN`` in that case.
    """
    provider = CountingEntropyProvider(
        prepare_discrete_data(data, device=device),
        estimator=entropy_estimator,
        count_mode="auto",
    )
    return provider.entropy(torch.as_tensor(variable_sets))


def information_measures(
    data: Any,
    variable_sets: Any,
    *,
    entropy_estimator: str = "empirical",
    device: torch.device | str | None = None,
    local_values: bool = False,
    max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
) -> InformationMeasures:
    """Calculate higher-order information measures for selected variable sets.

    The returned object provides total correlation, dual total correlation,
    O-information, and S-information through descriptive properties rather than
    requiring users to remember column positions. Sample-resolved values are
    optional and are not calculated by default.

    Parameters
    ----------
    data
        Binary observations or a prepared :class:`DiscreteData` object.
    variable_sets
        Variable indices with shape ``[sets, order]`` or one set with shape
        ``[order]``.
    entropy_estimator
        Global entropy estimator used consistently for every required subset
        entropy. Supported choices are ``"empirical"``, ``"miller_madow"``,
        ``"schurmann"``, ``"shrinkage"``, ``"chao_shen"``,
        ``"pitman_yor"``, and ``"ansb"``.
    device
        Optional Torch device on which calculations should be performed.
    local_values
        If ``True``, also calculate TC, DTC, O-information, and S-information
        for every observation. Local values are currently defined only for the
        empirical and Miller-Madow entropy conventions. Other estimators raise
        :class:`NotImplementedError` rather than imposing an arbitrary pointwise
        decomposition.
    max_local_values_per_batch
        Memory-control target for local calculations, expressed as the maximum
        approximate product ``variable sets × total samples`` processed at once.
        Explicit variable sets are automatically subdivided when necessary.
        Reducing this value lowers peak working memory; it does not alter the
        returned scientific values.

    Returns
    -------
    InformationMeasures
        Named access to the four global measures, each with shape
        ``[variable_sets, datasets]``. When ``local_values=True``, ``.local``
        contains one ``[variable_sets, samples_d]`` view per dataset and measure.

    Notes
    -----
    TC, DTC, O-information, and S-information are always derived from shared
    subset entropies rather than separate estimation pipelines. ANSB has the
    asymptotic requirement ``N/Q -> 0``; for binary singleton marginals
    ``Q=2``, this requirement generally fails. ANSB-based multivariate measures
    should therefore be interpreted only when the assumptions of every entropy
    term involved are scientifically defensible.

    Local arrays necessarily scale with the number of observations. dTHOI
    controls temporary work in batches and does not persistently cache local
    arrays, but the requested final local result still occupies
    ``O(variable_sets × samples × measures)`` memory.
    """
    output = nplets_measures(
        prepare_discrete_data(data, device=device),
        torch.as_tensor(variable_sets),
        estimator=entropy_estimator,
        count_mode="auto",
        local_values=local_values,
        max_local_values_per_batch=max_local_values_per_batch,
    )
    if isinstance(output, tuple):
        values, local = output
        return InformationMeasures(values, LocalInformationMeasures(local))
    return InformationMeasures(output)


def analyze_orders(
    data: Any,
    *,
    min_order: int = 3,
    max_order: int | None = None,
    entropy_estimator: str = "empirical",
    sets_per_batch: int = 4096,
    device: torch.device | str | None = None,
    local_values: bool = False,
    max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
) -> list[InteractionResults]:
    """Analyze every variable combination across a range of interaction orders.

    Parameters
    ----------
    data
        Binary observations or a prepared :class:`DiscreteData` object.
    min_order
        Smallest number of variables considered jointly.
    max_order
        Largest number of variables considered jointly. By default, all
        available variables are allowed.
    entropy_estimator
        Global entropy estimator reused across all interaction orders. Supported
        choices are ``"empirical"``, ``"miller_madow"``, ``"schurmann"``,
        ``"shrinkage"``, ``"chao_shen"``, ``"pitman_yor"``, and ``"ansb"``.
    sets_per_batch
        Maximum number of variable sets evaluated together. The default is
        intended to work well for routine global analyses.
    device
        Optional Torch device on which calculations should be performed.
    local_values
        If ``True``, also return sample-resolved values for all four measures.
        Local values are currently defined only for empirical and Miller-Madow
        entropy estimation.
    max_local_values_per_batch
        Memory-control target for local calculations, expressed as
        ``variable sets × total samples``. With local values enabled, dTHOI
        automatically reduces the effective ``sets_per_batch`` when the sample
        count would otherwise make a generated batch too large.

    Returns
    -------
    list[InteractionResults]
        Results grouped into memory-sized pieces. Each item states the
        interaction order, corresponding variable sets, and named information
        measures. Local arrays, when requested, are moved to CPU with each
        completed batch before the next batch is evaluated.

    Notes
    -----
    Batching controls peak working and accelerator memory, not the total size of
    the returned Python list. Exhaustive local analyses still scale with the
    total number of variable sets times the number of samples. The same ANSB
    marginal-regime limitation described in :func:`information_measures`
    applies to exhaustive analyses.
    """
    prepared = prepare_discrete_data(data, device=device)

    def collect(
        subsets: torch.Tensor,
        measures: MeasureOutput,
        order: int,
        batch_index: int,
    ) -> InteractionResults:
        """Convert one internal result batch to the user-facing representation."""
        del batch_index
        subsets_cpu = subsets.detach().cpu()
        if isinstance(measures, tuple):
            values, local = measures
            information = InformationMeasures(
                values.detach().cpu(),
                LocalInformationMeasures(
                    tuple(dataset_values.detach().cpu() for dataset_values in local)
                ),
            )
        else:
            information = InformationMeasures(measures.detach().cpu())
        return InteractionResults(
            order=order,
            variable_sets=subsets_cpu,
            information=information,
        )

    return multi_order_measures(
        prepared,
        min_order=min_order,
        max_order=max_order,
        estimator=entropy_estimator,
        count_mode="auto",
        batch_size=sets_per_batch,
        batch_collector=collect,
        local_values=local_values,
        max_local_values_per_batch=max_local_values_per_batch,
    )
