"""Conceptual integration contracts for predictive-interaction discovery.

This module intentionally defines interfaces only. Numerical implementations
belong in dTHOI or replaceable research adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class CandidateSet:
    """One structurally generated variable set.

    Parameters
    ----------
    variables
        Zero-based variable indices defining the set.
    sources
        Provenance labels such as ``forward_low_omega`` or
        ``reference_random``. Multiple labels are allowed after deduplication.
    omega_bits
        O-information of the set in bits when already computed.
    fold
        Outer-fold identifier when candidate generation is cross-fitted.
    """

    variables: tuple[int, ...]
    sources: tuple[str, ...]
    omega_bits: float | None = None
    fold: int | None = None

    @property
    def order(self) -> int:
        """Return the number of variables in the set."""
        return len(self.variables)


@dataclass(frozen=True)
class PredictiveScore:
    """Matched out-of-sample predictive scores for one variable set.

    Parameters
    ----------
    variables
        Zero-based variable indices defining the evaluated set.
    full_score
        Out-of-sample score from the flexible predictive model.
    additive_score
        Out-of-sample score from the matched no-interaction model.
    fold
        Evaluation fold identifier when scores are fold-specific.
    """

    variables: tuple[int, ...]
    full_score: float
    additive_score: float
    fold: int | None = None

    @property
    def interaction_gain(self) -> float:
        """Return flexible minus additive predictive performance."""
        return self.full_score - self.additive_score


class StructuralCandidateGenerator(Protocol):
    """Generate X-only structural candidates and same-order references.

    Future dTHOI integration point
    ------------------------------
    The intended implementation is a batched, Torch-first forward/backward
    O-information greedy search supplied by dTHOI. This branch should adapt to
    that API rather than duplicate its numerical implementation.
    """

    def generate(
        self,
        data: object,
        *,
        max_order: int,
        n_reference_sets: int,
        fold: int | None = None,
    ) -> Sequence[CandidateSet]:
        """Return random references plus low/high-O-information candidates."""
        ...


class PredictiveEvaluator(Protocol):
    """Evaluate matched flexible and additive models on supplied sets."""

    def evaluate(
        self,
        data: object,
        outcome: object,
        candidates: Sequence[CandidateSet],
        *,
        folds: object,
        weights: object | None = None,
    ) -> Sequence[PredictiveScore]:
        """Return out-of-sample full/additive scores for every candidate."""
        ...


class SemanticInterpreter(Protocol):
    """Optional downstream semantic interpretation of discovered candidates."""

    def interpret(
        self,
        data: object,
        outcome: object,
        candidates: Sequence[CandidateSet],
        *,
        folds: object,
        weights: object | None = None,
    ) -> object:
        """Return interpretable representations and their stability evidence."""
        ...
