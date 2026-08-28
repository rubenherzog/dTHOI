"""User-facing API for dTHOI.

The public API is intentionally phrased in scientific terms. Implementation
concepts such as counting strategies, entropy caches, estimator families, and
execution providers remain internal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .batch import multi_order_measures
from .data import DiscreteData, prepare_discrete_data
from .entropy.base import CountingEntropyProvider
from .measures.core import DEFAULT_MAX_LOCAL_VALUES_PER_BATCH
from .measures.definitions import MeasureOutput
from .measures.dispatch import measures_for_sets, resolve_public_estimator_argument


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
        ``"chao_shen"``, ``"pitman_yor"``, and ``"ansb"``.
    device
        Optional Torch device on which calculations should be performed.

    Returns
    -------
    torch.Tensor
        Entropy in bits with shape ``[variable_sets, datasets]``.

    Notes
    -----
    ``estimate_entropy`` accepts only estimators that define a standalone
    ``variable set -> entropy`` operation. Complete-joint coherent estimators
    such as ``"nsb"`` are selected through :func:`information_measures`, where
    their parent-set context is available.
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
    estimator: str | None = None,
    entropy_estimator: str | None = None,
    device: torch.device | str | None = None,
    local_values: bool = False,
    max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
) -> InformationMeasures:
    """Calculate TC, DTC, O-information, and S-information for variable sets.

    One ``estimator`` parameter selects the statistical method regardless of its
    internal family. Entropy-based methods compose shared subset entropies;
    coherent methods fit one complete-joint posterior and derive every required
    marginal from that same model. Both routes return the same scientific result
    object and use the same dTHOI definitions of TC, DTC, O-information, and
    S-information.

    Parameters
    ----------
    data
        Binary observations or a prepared :class:`DiscreteData` object.
    variable_sets
        Variable indices with shape ``[sets, order]`` or one set ``[order]``.
    estimator
        Information estimator. Supported entropy-based choices are
        ``"empirical"``, ``"miller_madow"``, ``"schurmann"``, ``"shrinkage"``,
        ``"chao_shen"``, ``"pitman_yor"``, and ``"ansb"``. Coherent
        complete-joint choices are ``"dirichlet_a1"``, ``"dirichlet_eb"``, and
        ``"nsb"``. The default is ``"empirical"``.
    entropy_estimator
        Backward-compatible alias for ``estimator``. New code should use
        ``estimator``. Supplying both arguments raises :class:`ValueError`.
    device
        Optional Torch device on which calculations should be performed.
    local_values
        If ``True``, also calculate sample-resolved values. Local values are
        currently defined for ``"empirical"``, ``"miller_madow"``,
        ``"schurmann"``, and ``"chao_shen"``. Estimators without an explicit
        pointwise convention raise :class:`NotImplementedError`.
    max_local_values_per_batch
        Memory-control target for local calculations, expressed as the maximum
        approximate product ``variable sets × total samples`` processed at once.

    Returns
    -------
    InformationMeasures
        Named access to the four global measures in bits. Each has shape
        ``[variable_sets, datasets]``. Optional local values remain separated by
        dataset so unequal sample counts are never padded or merged.

    Notes
    -----
    dTHOI always uses ``O = TC - DTC`` and ``S = TC + DTC``. For coherent
    Dirichlet estimators, all entropy terms are posterior expectations under one
    parent joint model, so TC and DTC remain non-negative apart from numerical
    roundoff. Avoiding ``2**order`` state-space allocation removes an exponential
    memory requirement but does not make severely undersampled high-order
    distributions statistically identifiable.
    """
    selected = resolve_public_estimator_argument(
        estimator=estimator,
        entropy_estimator=entropy_estimator,
    )
    output = measures_for_sets(
        prepare_discrete_data(data, device=device),
        torch.as_tensor(variable_sets),
        estimator=selected,
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
    estimator: str | None = None,
    entropy_estimator: str | None = None,
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
    estimator
        Information estimator selected from the same choices accepted by
        :func:`information_measures`. The default is ``"empirical"``.
    entropy_estimator
        Backward-compatible alias for ``estimator``. New code should use
        ``estimator``; supplying both arguments raises :class:`ValueError`.
    sets_per_batch
        Maximum number of generated variable sets per returned batch. Internal
        estimators may reduce the working batch further to bound memory.
    device
        Optional Torch device on which calculations should be performed.
    local_values
        If ``True``, also return sample-resolved values for estimators that
        define them.
    max_local_values_per_batch
        Memory-control target for local calculations, expressed as
        ``variable sets × total samples``.

    Returns
    -------
    list[InteractionResults]
        Results grouped into bounded pieces. Each item states the interaction
        order, variable sets, and named information measures.
    """
    prepared = prepare_discrete_data(data, device=device)
    selected = resolve_public_estimator_argument(
        estimator=estimator,
        entropy_estimator=entropy_estimator,
    )

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
        estimator=selected,
        count_mode="auto",
        batch_size=sets_per_batch,
        batch_collector=collect,
        local_values=local_values,
        max_local_values_per_batch=max_local_values_per_batch,
    )
