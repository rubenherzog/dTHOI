"""Coherent Bayesian estimators for higher-order information measures.

Each complete variable set is assigned one symmetric Dirichlet model. Joint,
singleton, and leave-one-out entropy terms are then evaluated as marginals of
that same posterior model. For interaction orders up to 63, the implementation
reuses dTHOI's batched exact state encoding and sparse observed-state counting
primitives. Wider sets use an exact observed-row fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Literal

import torch

from ..data import DiscreteInput, prepare_discrete_data
from ..entropy.counting import (
    SparseCounts,
    encode_binary_rows,
    encode_binary_states,
    mask_binary_rows,
    sparse_counts_from_codes,
)
from ..subsets import canonicalize_subsets, deduplicate_subsets, leave_one_out_subsets
from .definitions import combine_global_terms

CoherentEstimator = Literal["dirichlet_a1", "dirichlet_eb", "nsb"]
COHERENT_ESTIMATORS = frozenset({"dirichlet_a1", "dirichlet_eb", "nsb"})

_LOG_2 = math.log(2.0)
_GRID_POINTS = 72
_EDGE_MASS_THRESHOLD = 1.0e-4


@dataclass(frozen=True)
class _WideCountSummary:
    """Observed counts required for one arbitrary-width complete variable set.

    Parameters
    ----------
    order
        Number of variables in the complete set.
    samples
        Number of observations.
    full
        Positive complete-state counts.
    leave_one_out
        Positive counts for every one-variable deletion.
    singletons
        Positive counts for every singleton marginal.
    """

    order: int
    samples: int
    full: torch.Tensor
    leave_one_out: tuple[torch.Tensor, ...]
    singletons: tuple[torch.Tensor, ...]


def normalize_coherent_estimator(estimator: str) -> CoherentEstimator:
    """Validate and normalize a coherent information-measure estimator name."""
    key = estimator.lower().replace("-", "_")
    if key not in COHERENT_ESTIMATORS:
        supported = ", ".join(sorted(COHERENT_ESTIMATORS))
        raise ValueError(
            f"Unknown coherent information estimator {estimator!r}. "
            f"Supported choices are: {supported}."
        )
    return key  # type: ignore[return-value]


def _segment_sum(values: torch.Tensor, row_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sum values independently for packed sparse histograms."""
    result = torch.zeros(
        (batch_size, *values.shape[1:]), dtype=values.dtype, device=values.device
    )
    return result.index_add_(0, row_ids, values)


def _observed_state_counts(counts: SparseCounts) -> torch.Tensor:
    """Return the number of occupied states in each packed histogram."""
    return torch.bincount(counts.row_ids, minlength=counts.batch_size).to(
        device=counts.values.device, dtype=torch.float64
    )


def _histogram_totals(counts: SparseCounts) -> torch.Tensor:
    """Return total observations in each packed histogram."""
    values = counts.values.to(dtype=torch.float64)
    return _segment_sum(values, counts.row_ids, counts.batch_size)


def _inverse_cardinality(order: int) -> float:
    """Return ``2**(-order)`` without constructing a binary state space."""
    return math.ldexp(1.0, -order)


def _expected_entropy_grid(
    counts: SparseCounts,
    order: int,
    concentrations: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate posterior-mean Shannon entropy over a concentration grid.

    For total symmetric Dirichlet concentration :math:`A` and binary subset
    order :math:`r`, every state receives prior mass :math:`\alpha=A/2^r`.
    Unobserved states are aggregated analytically, so only positive observed
    counts are stored. Returned entropy is in bits.
    """
    a = concentrations.to(device=counts.values.device, dtype=torch.float64)
    values = counts.values.to(dtype=torch.float64)
    alpha = a * _inverse_cardinality(order)
    beta = values.unsqueeze(1) + alpha.unsqueeze(0)
    observed_term = _segment_sum(
        beta * torch.special.digamma(beta + 1.0), counts.row_ids, counts.batch_size
    )
    occupied = _observed_state_counts(counts).unsqueeze(1)
    unseen_mass = (a.unsqueeze(0) - occupied * alpha.unsqueeze(0)).clamp_min(0.0)
    unseen_term = unseen_mass * torch.special.digamma(alpha + 1.0).unsqueeze(0)
    total = _histogram_totals(counts).unsqueeze(1) + a.unsqueeze(0)
    return (
        torch.special.digamma(total + 1.0) - (observed_term + unseen_term) / total
    ) / _LOG_2


def _log_evidence_grid(
    counts: SparseCounts,
    order: int,
    concentrations: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate complete-joint Dirichlet-multinomial log evidence over ``A``.

    Terms that do not depend on :math:`A` are omitted because EB selection and
    NSB weighting use only relative evidence.
    """
    a = concentrations.to(device=counts.values.device, dtype=torch.float64)
    values = counts.values.to(dtype=torch.float64)
    alpha = a * _inverse_cardinality(order)
    observed_terms = (
        torch.lgamma(values.unsqueeze(1) + alpha.unsqueeze(0))
        - torch.lgamma(alpha).unsqueeze(0)
    )
    summed = _segment_sum(observed_terms, counts.row_ids, counts.batch_size)
    totals = _histogram_totals(counts).unsqueeze(1)
    return torch.lgamma(a).unsqueeze(0) - torch.lgamma(totals + a.unsqueeze(0)) + summed


def _nsb_log_hyperprior(concentrations: torch.Tensor, order: int) -> torch.Tensor:
    """Return the NSB entropy-flat log hyperprior including the log-A Jacobian."""
    a = concentrations.to(dtype=torch.float64)
    inv_q = _inverse_cardinality(order)
    alpha = a * inv_q
    derivative = (
        torch.special.polygamma(1, a + 1.0)
        - inv_q * torch.special.polygamma(1, alpha + 1.0)
    ).clamp_min(torch.finfo(torch.float64).tiny)
    return torch.log(derivative) + torch.log(a)


def _grid_edge_flags(
    evidence: torch.Tensor,
    concentrations: torch.Tensor,
    order: int,
    estimator: CoherentEstimator,
) -> tuple[bool, bool]:
    """Detect whether EB optima or NSB posterior mass touch grid boundaries."""
    if estimator == "dirichlet_eb":
        maxima = torch.argmax(evidence, dim=1)
        return (
            bool(torch.any(maxima == 0)),
            bool(torch.any(maxima == evidence.shape[1] - 1)),
        )

    log_weights = evidence + _nsb_log_hyperprior(concentrations, order).unsqueeze(0)
    weights = torch.softmax(
        log_weights - log_weights.max(dim=1, keepdim=True).values,
        dim=1,
    )
    edge_width = min(2, weights.shape[1])
    return (
        bool(torch.any(weights[:, :edge_width].sum(dim=1) > _EDGE_MASS_THRESHOLD)),
        bool(torch.any(weights[:, -edge_width:].sum(dim=1) > _EDGE_MASS_THRESHOLD)),
    )


def _concentration_grid(
    full_counts: SparseCounts,
    order: int,
    estimator: CoherentEstimator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic concentration grid and its joint evidence.

    The grid depends only on sample count and interaction order, not on which
    other parent sets happen to share the execution batch. This invariance is
    required so batched and setwise execution return the same scientific value.
    A boundary diagnostic is emitted rather than changing the grid for the
    whole batch.
    """
    if estimator == "dirichlet_a1":
        grid = torch.ones(1, dtype=torch.float64, device=full_counts.values.device)
        evidence = torch.empty(
            (full_counts.batch_size, 0),
            dtype=torch.float64,
            device=full_counts.values.device,
        )
        return grid, evidence

    samples = float(_histogram_totals(full_counts)[0].item())
    upper = max(1.0e4, 100.0 * samples)
    grid = torch.logspace(
        -4.0,
        math.log10(upper),
        _GRID_POINTS,
        dtype=torch.float64,
        device=full_counts.values.device,
    )
    evidence = _log_evidence_grid(full_counts, order, grid)
    lower_edge, upper_edge = _grid_edge_flags(evidence, grid, order, estimator)
    if lower_edge or upper_edge:
        warnings.warn(
            "The coherent estimator retains an optimum or posterior mass at a "
            "concentration-grid boundary; the result may be grid-sensitive.",
            RuntimeWarning,
            stacklevel=3,
        )
    return grid, evidence


def _select_measure_grid(
    measure_grid: torch.Tensor,
    evidence: torch.Tensor,
    concentrations: torch.Tensor,
    order: int,
    estimator: CoherentEstimator,
) -> torch.Tensor:
    """Select EB values or integrate NSB values from a coherent measure grid."""
    if estimator == "dirichlet_a1":
        return measure_grid[:, 0]
    if estimator == "dirichlet_eb":
        rows = torch.arange(measure_grid.shape[0], device=measure_grid.device)
        return measure_grid[rows, torch.argmax(evidence, dim=1)]
    log_weights = evidence + _nsb_log_hyperprior(concentrations, order).unsqueeze(0)
    weights = torch.softmax(
        log_weights - log_weights.max(dim=1, keepdim=True).values,
        dim=1,
    )
    return torch.sum(weights.unsqueeze(2) * measure_grid, dim=1)


def _sparse_counts_for_subsets(
    dataset: torch.Tensor,
    subsets: torch.Tensor,
    masks: list[int],
    row_codes: torch.Tensor | None,
) -> SparseCounts:
    """Count observed states for a fixed-order subset batch using core primitives."""
    if subsets.shape[1] > 63:
        raise NotImplementedError("Packed batched counting requires subset order <= 63.")
    codes = (
        mask_binary_rows(row_codes, masks)
        if row_codes is not None
        else encode_binary_states(dataset, subsets)
    )
    packed = sparse_counts_from_codes(codes)
    assert isinstance(packed, SparseCounts)
    return packed


def _batched_dataset_measures(
    dataset: torch.Tensor,
    parent_subsets: torch.Tensor,
    parent_masks: list[int],
    estimator: CoherentEstimator,
) -> torch.Tensor:
    """Estimate coherent measures for one dataset and a bounded parent-set batch."""
    batch_size, order = parent_subsets.shape
    row_codes = encode_binary_rows(dataset) if dataset.shape[1] <= 63 else None
    full_counts = _sparse_counts_for_subsets(
        dataset, parent_subsets, parent_masks, row_codes
    )
    concentrations, evidence = _concentration_grid(full_counts, order, estimator)
    h_joint = _expected_entropy_grid(full_counts, order, concentrations)

    singleton_terms = parent_subsets.reshape(batch_size * order, 1)
    unique_singletons, singleton_inverse, singleton_masks = deduplicate_subsets(
        singleton_terms
    )
    singleton_counts = _sparse_counts_for_subsets(
        dataset, unique_singletons, singleton_masks, row_codes
    )
    singleton_entropy = _expected_entropy_grid(singleton_counts, 1, concentrations)
    h_single = singleton_entropy.index_select(0, singleton_inverse).reshape(
        batch_size, order, -1
    ).sum(dim=1)

    loo_terms = leave_one_out_subsets(parent_subsets)
    unique_loo, loo_inverse, loo_masks = deduplicate_subsets(loo_terms)
    loo_counts = _sparse_counts_for_subsets(
        dataset, unique_loo, loo_masks, row_codes
    )
    loo_entropy = _expected_entropy_grid(loo_counts, order - 1, concentrations)
    h_loo = loo_entropy.index_select(0, loo_inverse).reshape(
        batch_size, order, -1
    ).sum(dim=1)

    measure_grid = combine_global_terms(h_joint, h_single, h_loo, order)
    return _select_measure_grid(
        measure_grid, evidence, concentrations, order, estimator
    )


def _weighted_observed_counts(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Aggregate complete-state multiplicities for identical arbitrary-width rows."""
    unique_rows, inverse = torch.unique(rows, dim=0, return_inverse=True)
    counts = torch.zeros(
        unique_rows.shape[0], dtype=torch.float64, device=weights.device
    )
    counts.index_add_(0, inverse, weights.to(dtype=torch.float64))
    return counts


def _summarize_wide_counts(data: torch.Tensor) -> _WideCountSummary:
    """Build exact joint and induced marginal counts for a set wider than 63 bits."""
    states, full_counts_int = torch.unique(data, dim=0, return_counts=True)
    full_counts = full_counts_int.to(dtype=torch.float64)
    order = int(data.shape[1])
    samples = int(data.shape[0])

    state_values = states.to(dtype=torch.float64)
    ones = state_values.transpose(0, 1).matmul(full_counts)
    total = torch.tensor(float(samples), dtype=torch.float64, device=data.device)
    singletons: list[torch.Tensor] = []
    for ones_count in ones.unbind():
        counts = torch.stack((total - ones_count, ones_count))
        singletons.append(counts[counts > 0.0])

    variable_positions = torch.arange(order, device=data.device)
    leave_one_out: list[torch.Tensor] = []
    for removed in range(order):
        kept = variable_positions != removed
        leave_one_out.append(
            _weighted_observed_counts(states[:, kept], full_counts)
        )

    return _WideCountSummary(
        order,
        samples,
        full_counts,
        tuple(leave_one_out),
        tuple(singletons),
    )


def _as_single_sparse(counts: torch.Tensor) -> SparseCounts:
    """Wrap one positive-count vector in the common packed histogram representation."""
    return SparseCounts(
        values=counts,
        row_ids=torch.zeros(
            counts.numel(), dtype=torch.int64, device=counts.device
        ),
        batch_size=1,
    )


def _wide_summary_measures(
    summary: _WideCountSummary,
    estimator: CoherentEstimator,
) -> torch.Tensor:
    """Estimate one arbitrary-width parent set using common posterior equations."""
    full = _as_single_sparse(summary.full)
    concentrations, evidence = _concentration_grid(
        full, summary.order, estimator
    )
    h_joint = _expected_entropy_grid(full, summary.order, concentrations)
    h_single = torch.stack(
        [
            _expected_entropy_grid(
                _as_single_sparse(counts), 1, concentrations
            )[0]
            for counts in summary.singletons
        ],
        dim=0,
    ).sum(dim=0, keepdim=True)
    h_loo = torch.stack(
        [
            _expected_entropy_grid(
                _as_single_sparse(counts), summary.order - 1, concentrations
            )[0]
            for counts in summary.leave_one_out
        ],
        dim=0,
    ).sum(dim=0, keepdim=True)
    measure_grid = combine_global_terms(
        h_joint, h_single, h_loo, summary.order
    )
    return _select_measure_grid(
        measure_grid,
        evidence,
        concentrations,
        summary.order,
        estimator,
    )[0]


def coherent_measures(
    X: DiscreteInput,
    subsets: torch.Tensor,
    *,
    estimator: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Estimate coherent TC, DTC, O-information, and S-information in bits.

    Parameters
    ----------
    X
        Binary observations accepted by :func:`dthoi.prepare_data`.
    subsets
        Fixed-order variable sets with shape ``[sets, order]`` or one set
        ``[order]``. Interaction order must be at least two.
    estimator
        ``"dirichlet_a1"``, ``"dirichlet_eb"``, or ``"nsb"``.
    device
        Optional destination Torch device.

    Returns
    -------
    torch.Tensor
        TC, DTC, O-information, and S-information in bits with shape
        ``[variable_sets, datasets, 4]``.

    Notes
    -----
    A single symmetric Dirichlet model is placed on each complete variable set.
    Marginalization preserves total concentration ``A``. The nominal
    ``2**order`` state space is never allocated; order ``<=63`` uses batched
    packed counting and wider sets use an exact observed-row fallback.
    """
    resolved = normalize_coherent_estimator(estimator)
    prepared = prepare_discrete_data(X, device=device)
    canonical = canonicalize_subsets(
        subsets, prepared.n_variables
    ).to(prepared.device)
    if canonical.shape[1] < 2:
        raise ValueError("Higher-order information measures require order >= 2.")

    unique_subsets, inverse, unique_masks = deduplicate_subsets(canonical)
    unique_subsets = unique_subsets.to(prepared.device)
    inverse = inverse.to(prepared.device)
    order = unique_subsets.shape[1]
    unique_output = torch.empty(
        (unique_subsets.shape[0], prepared.n_datasets, 4),
        dtype=torch.float64,
        device=prepared.device,
    )

    if order <= 63:
        for dataset_index, dataset in enumerate(prepared.datasets):
            unique_output[:, dataset_index] = _batched_dataset_measures(
                dataset,
                unique_subsets,
                unique_masks,
                resolved,
            )
    else:
        for dataset_index, dataset in enumerate(prepared.datasets):
            for row, subset in enumerate(unique_subsets):
                selected = dataset.index_select(1, subset)
                unique_output[row, dataset_index] = _wide_summary_measures(
                    _summarize_wide_counts(selected),
                    resolved,
                )

    return unique_output.index_select(0, inverse)
