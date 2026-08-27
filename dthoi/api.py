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
from .measures.core import nplets_measures


@dataclass(frozen=True)
class InformationMeasures:
    """Higher-order information measures for one or more variable sets.

    Parameters
    ----------
    values
        Tensor whose final dimension contains, in order, total correlation,
        dual total correlation, O-information, and S-information. For the
        standard dTHOI API the shape is ``[variable_sets, datasets, 4]``.

    Notes
    -----
    The named properties are views of ``values`` and therefore do not duplicate
    the underlying data.
    """

    values: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the measure dimension."""
        if self.values.ndim < 1 or self.values.shape[-1] != 4:
            raise ValueError("The final dimension must contain the four dTHOI measures.")

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
        Higher-order information measures for those variable sets.
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


def entropy(
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
        Entropy estimator. ``"empirical"`` uses observed state frequencies;
        ``"miller_madow"`` applies the Miller-Madow finite-sample correction.
    device
        Optional Torch device on which calculations should be performed.

    Returns
    -------
    torch.Tensor
        Entropy in bits with shape ``[variable_sets, datasets]``.
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
) -> InformationMeasures:
    """Calculate higher-order information measures for selected variable sets.

    The returned object provides total correlation, dual total correlation,
    O-information, and S-information through descriptive properties rather than
    requiring users to remember column positions.

    Parameters
    ----------
    data
        Binary observations or a prepared :class:`DiscreteData` object.
    variable_sets
        Variable indices with shape ``[sets, order]`` or one set with shape
        ``[order]``.
    entropy_estimator
        Entropy estimator. Supported public choices are ``"empirical"`` and
        ``"miller_madow"``.
    device
        Optional Torch device on which calculations should be performed.

    Returns
    -------
    InformationMeasures
        Named access to the four measures. Each measure has shape
        ``[variable_sets, datasets]``.
    """
    values = nplets_measures(
        prepare_discrete_data(data, device=device),
        torch.as_tensor(variable_sets),
        estimator=entropy_estimator,
        count_mode="auto",
    )
    return InformationMeasures(values)


def analyze_orders(
    data: Any,
    *,
    min_order: int = 3,
    max_order: int | None = None,
    entropy_estimator: str = "empirical",
    sets_per_batch: int = 4096,
    device: torch.device | str | None = None,
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
        Entropy estimator. Supported public choices are ``"empirical"`` and
        ``"miller_madow"``.
    sets_per_batch
        Maximum number of variable sets evaluated together. The default is
        intended to work well for routine analyses; reducing it lowers peak
        memory use.
    device
        Optional Torch device on which calculations should be performed.

    Returns
    -------
    list[InteractionResults]
        Results grouped into memory-sized pieces. Each item states the
        interaction order, the corresponding variable sets, and their named
        information measures.
    """
    prepared = prepare_discrete_data(data, device=device)

    def collect(
        subsets: torch.Tensor,
        measures: torch.Tensor,
        order: int,
        batch_index: int,
    ) -> InteractionResults:
        """Convert one internal result batch to the user-facing representation."""
        del batch_index
        return InteractionResults(
            order=order,
            variable_sets=subsets.detach().cpu(),
            information=InformationMeasures(measures.detach().cpu()),
        )

    return multi_order_measures(
        prepared,
        min_order=min_order,
        max_order=max_order,
        estimator=entropy_estimator,
        count_mode="auto",
        batch_size=sets_per_batch,
        batch_collector=collect,
    )
