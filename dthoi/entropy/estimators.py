from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch

_LOG_2 = math.log(2.0)


def _prepare_dense_counts(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert counts to float64 and return validated row totals."""
    counts_f = counts.to(dtype=torch.float64)
    totals = counts_f.sum(dim=-1)
    if bool(torch.any(totals <= 0)):
        raise ValueError("Counts must contain at least one observation per row.")
    return counts_f, totals


def _plugin_entropy(counts_f: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    """Compute plug-in entropy from validated float64 counts and totals."""
    sum_c_log2_c = torch.special.xlogy(counts_f, counts_f).sum(dim=-1) / _LOG_2
    return torch.log2(totals) - sum_c_log2_c / totals


class CountEntropyEstimator(ABC):
    """Interface for entropy estimators operating on exact state counts."""

    name: str

    @abstractmethod
    def entropy_from_dense(self, counts: torch.Tensor) -> torch.Tensor:
        """Estimate one entropy per row of dense ``[B, Q]`` counts."""

    def entropy_from_sparse(self, counts: list[torch.Tensor]) -> torch.Tensor:
        """Estimate one entropy per observed-state count vector.

        Subclasses may override this method with a specialized sparse path. The
        default implementation reuses :meth:`entropy_from_dense` without
        changing the estimator definition.
        """
        values = [self.entropy_from_dense(row.unsqueeze(0))[0] for row in counts]
        if not values:
            return torch.empty(0, dtype=torch.float64)
        return torch.stack(values)


class PluginEstimator(CountEntropyEstimator):
    """Maximum-likelihood (plug-in) Shannon entropy estimator in bits."""

    name = "plugin"

    def entropy_from_dense(self, counts: torch.Tensor) -> torch.Tensor:
        """Estimate plug-in Shannon entropy from dense state counts."""
        counts_f, totals = _prepare_dense_counts(counts)
        return _plugin_entropy(counts_f, totals)


class MillerMadowEstimator(CountEntropyEstimator):
    """Miller-Madow finite-sample correction to plug-in entropy."""

    name = "miller_madow"

    def entropy_from_dense(self, counts: torch.Tensor) -> torch.Tensor:
        """Estimate Miller-Madow-corrected Shannon entropy from counts."""
        counts_f, totals = _prepare_dense_counts(counts)
        plugin = _plugin_entropy(counts_f, totals)
        observed = (counts_f > 0).sum(dim=-1).to(dtype=torch.float64)
        correction = (observed - 1.0) / (2.0 * totals * _LOG_2)
        return plugin + correction


def resolve_estimator(estimator: str | CountEntropyEstimator) -> CountEntropyEstimator:
    """Resolve a user estimator specification to an estimator instance.

    Parameters
    ----------
    estimator
        Registered estimator name or an existing :class:`CountEntropyEstimator`.

    Returns
    -------
    CountEntropyEstimator
        Resolved estimator instance.
    """
    if isinstance(estimator, CountEntropyEstimator):
        return estimator
    if not isinstance(estimator, str):
        raise TypeError("estimator must be a string or CountEntropyEstimator instance.")

    key = estimator.lower().replace("-", "_")
    if key in {"plugin", "empirical", "maximum_likelihood", "ml"}:
        return PluginEstimator()
    if key in {"miller_madow", "mm"}:
        return MillerMadowEstimator()
    raise ValueError(f"Unknown estimator: {estimator!r}.")
