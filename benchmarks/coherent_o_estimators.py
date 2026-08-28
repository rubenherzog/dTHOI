"""Scalable coherent Bayesian O-information estimators for research benchmarks.

The estimators in this module place one symmetric Dirichlet model on the full
joint distribution of a variable set. Every marginal entropy is therefore
computed from a marginal of the same posterior model. State spaces are never
materialized: only observed-state counts plus the scalar nominal cardinality
enter the formulas.

These functions are intentionally benchmark-local until their finite-sample
properties justify a public dTHOI API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

_LOG2 = math.log(2.0)


@dataclass(frozen=True)
class CountSummary:
    """Observed-state counts required for O-information of one variable set."""

    order: int
    samples: int
    full: np.ndarray
    leave_one_out: tuple[np.ndarray, ...]
    singletons: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CoherentOEstimate:
    """TC, DTC, O-information, and S-information in bits for one estimator."""

    total_correlation: float
    dual_total_correlation: float
    o_information: float
    s_information: float
    concentration: float


def _state_counts(data: np.ndarray) -> np.ndarray:
    """Return positive counts without materializing the nominal state space.

    Integer packing is used through order 63. Wider variable sets use packed
    byte rows so the statistical estimator itself is not restricted to a
    machine-word state key.
    """
    values = np.asarray(data, dtype=np.uint8)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must have non-empty shape [samples, variables].")
    order = values.shape[1]
    if order <= 63:
        weights = np.left_shift(np.uint64(1), np.arange(order, dtype=np.uint64))
        codes = values.astype(np.uint64, copy=False) @ weights
        _, counts = np.unique(codes, return_counts=True)
    else:
        packed = np.packbits(values, axis=1, bitorder="little")
        _, counts = np.unique(packed, axis=0, return_counts=True)
    return counts.astype(np.float64, copy=False)


def summarize_counts(data: np.ndarray) -> CountSummary:
    """Count the full set, all leave-one-out sets, and all single variables."""
    values = np.asarray(data, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("O-information requires data [samples, variables] with order >= 3.")
    order = values.shape[1]
    full = _state_counts(values)
    leave = tuple(_state_counts(np.delete(values, i, axis=1)) for i in range(order))
    singletons = tuple(_state_counts(values[:, i : i + 1]) for i in range(order))
    return CountSummary(order, int(values.shape[0]), full, leave, singletons)


def _inverse_cardinality(order: int) -> float:
    """Return 2**(-order) without constructing the state space."""
    return math.ldexp(1.0, -order)


def _expected_entropy_grid(
    counts: np.ndarray,
    order: int,
    samples: int,
    concentrations: torch.Tensor,
) -> torch.Tensor:
    """Posterior mean Shannon entropy under symmetric Dirichlet concentrations.

    ``concentrations`` contains the total Dirichlet concentration A. Each of the
    2**order nominal cells has prior mass A / 2**order. Unobserved cells are
    aggregated analytically rather than enumerated.
    """
    a = concentrations.to(dtype=torch.float64)
    inv_q = _inverse_cardinality(order)
    alpha = a * inv_q
    observed = torch.as_tensor(counts, dtype=torch.float64, device=a.device)
    beta = observed.unsqueeze(0) + alpha.unsqueeze(1)
    observed_term = (beta * torch.special.digamma(beta + 1.0)).sum(dim=1)
    unseen_mass = (a - float(len(counts)) * alpha).clamp_min(0.0)
    unseen_term = unseen_mass * torch.special.digamma(alpha + 1.0)
    total = float(samples) + a
    return (
        torch.special.digamma(total + 1.0)
        - (observed_term + unseen_term) / total
    ) / _LOG2


def _measure_grid(summary: CountSummary, concentrations: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return coherent TC, DTC, O, and S for every concentration in a grid."""
    full_h = _expected_entropy_grid(
        summary.full, summary.order, summary.samples, concentrations
    )
    singleton_h = torch.stack(
        [
            _expected_entropy_grid(counts, 1, summary.samples, concentrations)
            for counts in summary.singletons
        ],
        dim=0,
    ).sum(dim=0)
    leave_h = torch.stack(
        [
            _expected_entropy_grid(
                counts, summary.order - 1, summary.samples, concentrations
            )
            for counts in summary.leave_one_out
        ],
        dim=0,
    ).sum(dim=0)
    tc = singleton_h - full_h
    dtc = leave_h - float(summary.order - 1) * full_h
    return tc, dtc, tc - dtc, tc + dtc


def _log_evidence_grid(summary: CountSummary, concentrations: torch.Tensor) -> torch.Tensor:
    """Log Dirichlet-multinomial evidence up to A-independent constants."""
    a = concentrations.to(dtype=torch.float64)
    inv_q = _inverse_cardinality(summary.order)
    alpha = a * inv_q
    counts = torch.as_tensor(summary.full, dtype=torch.float64, device=a.device)
    terms = torch.lgamma(counts.unsqueeze(0) + alpha.unsqueeze(1))
    terms = terms - torch.lgamma(alpha).unsqueeze(1)
    return torch.lgamma(a) - torch.lgamma(a + float(summary.samples)) + terms.sum(dim=1)


def _concentration_grid(samples: int, points: int = 72) -> torch.Tensor:
    """Return a broad logarithmic grid for evidence and NSB integration."""
    upper = max(1.0e4, 100.0 * float(samples))
    return torch.logspace(-4.0, math.log10(upper), points, dtype=torch.float64)


def coherent_dirichlet_a1(summary: CountSummary) -> CoherentOEstimate:
    """Posterior-mean O-information under one coherent A=1 Dirichlet prior."""
    a = torch.tensor([1.0], dtype=torch.float64)
    tc, dtc, o, s = _measure_grid(summary, a)
    return CoherentOEstimate(float(tc[0]), float(dtc[0]), float(o[0]), float(s[0]), 1.0)


def coherent_dirichlet_empirical_bayes(summary: CountSummary) -> CoherentOEstimate:
    """Posterior-mean O-information with A chosen by joint marginal likelihood."""
    grid = _concentration_grid(summary.samples)
    evidence = _log_evidence_grid(summary, grid)
    index = int(torch.argmax(evidence))
    tc, dtc, o, s = _measure_grid(summary, grid[index : index + 1])
    return CoherentOEstimate(
        float(tc[0]), float(dtc[0]), float(o[0]), float(s[0]), float(grid[index])
    )


def coherent_nsb(summary: CountSummary) -> CoherentOEstimate:
    """Posterior-mean O-information under one coherent NSB-style mixture.

    A single symmetric Dirichlet concentration A governs the full joint and all
    of its marginals. The hyperprior is proportional to the derivative of the
    prior expected full-joint entropy with respect to A, as in the NSB
    construction. Numerical integration is one-dimensional on log(A).
    """
    grid = _concentration_grid(summary.samples)
    evidence = _log_evidence_grid(summary, grid)
    inv_q = _inverse_cardinality(summary.order)
    alpha = grid * inv_q
    entropy_derivative = (
        torch.special.polygamma(1, grid + 1.0)
        - inv_q * torch.special.polygamma(1, alpha + 1.0)
    ).clamp_min(torch.finfo(torch.float64).tiny)
    # The grid is uniform in log(A), hence the A Jacobian.
    log_weight = evidence + torch.log(entropy_derivative) + torch.log(grid)
    weights = torch.softmax(log_weight - torch.max(log_weight), dim=0)
    tc, dtc, o, s = _measure_grid(summary, grid)
    mean_a = float(torch.sum(weights * grid))
    return CoherentOEstimate(
        float(torch.sum(weights * tc)),
        float(torch.sum(weights * dtc)),
        float(torch.sum(weights * o)),
        float(torch.sum(weights * s)),
        mean_a,
    )


def all_coherent_estimators(data: np.ndarray) -> dict[str, CoherentOEstimate]:
    """Evaluate the three research estimators while reusing one count summary."""
    summary = summarize_counts(data)
    return {
        "coherent_dirichlet_a1": coherent_dirichlet_a1(summary),
        "coherent_dirichlet_eb": coherent_dirichlet_empirical_bayes(summary),
        "coherent_nsb": coherent_nsb(summary),
    }
