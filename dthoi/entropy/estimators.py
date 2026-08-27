from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch


class CountEntropyEstimator(ABC):
    """Minimal interface for entropy estimators operating on state counts."""

    name: str

    @abstractmethod
    def entropy_from_dense(self, counts: torch.Tensor) -> torch.Tensor:
        """Estimate entropy for dense counts with shape ``[B, Q]``."""

    def entropy_from_sparse(self, counts: list[torch.Tensor]) -> torch.Tensor:
        values = [self.entropy_from_dense(c.unsqueeze(0))[0] for c in counts]
        if not values:
            return torch.empty(0, dtype=torch.float64)
        return torch.stack(values)


class PluginEstimator(CountEntropyEstimator):
    name = "plugin"

    def entropy_from_dense(self, counts: torch.Tensor) -> torch.Tensor:
        counts = counts.to(dtype=torch.float64)
        totals = counts.sum(dim=-1)
        if bool(torch.any(totals <= 0)):
            raise ValueError("Counts must contain at least one observation per row.")

        # H = log2(T) - (1/T) * sum_c c log2(c). This is algebraically
        # identical to the probability form while avoiding large probability
        # and term tensors in the dense path.
        sum_c_log2_c = torch.special.xlogy(counts, counts).sum(dim=-1) / math.log(2.0)
        return torch.log2(totals) - sum_c_log2_c / totals


class MillerMadowEstimator(CountEntropyEstimator):
    """Plug-in entropy with the Miller-Madow finite-sample correction."""

    name = "miller_madow"

    def entropy_from_dense(self, counts: torch.Tensor) -> torch.Tensor:
        counts_f = counts.to(dtype=torch.float64)
        totals = counts_f.sum(dim=-1)
        if bool(torch.any(totals <= 0)):
            raise ValueError("Counts must contain at least one observation per row.")

        plugin = PluginEstimator().entropy_from_dense(counts_f)
        observed = (counts_f > 0).sum(dim=-1).to(dtype=torch.float64)
        correction = (observed - 1.0) / (2.0 * totals * math.log(2.0))
        return plugin + correction


def resolve_estimator(estimator: str | CountEntropyEstimator) -> CountEntropyEstimator:
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
