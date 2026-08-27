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


def _plugin_local_from_dense(
    counts_f: torch.Tensor,
    totals: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Return empirical surprisal for dense state codes in sample order."""
    if codes.ndim != 2 or codes.shape[0] != counts_f.shape[0]:
        raise ValueError("codes must have shape [B, T] matching the count rows.")
    observed_counts = torch.gather(counts_f, 1, codes.to(dtype=torch.long))
    return torch.log2(totals).unsqueeze(1) - torch.log2(observed_counts)


def _plugin_local_from_sparse(
    counts: list[torch.Tensor],
    inverse: list[torch.Tensor],
) -> torch.Tensor:
    """Return empirical surprisal from sparse counts and sample-state mappings."""
    if len(counts) != len(inverse):
        raise ValueError("counts and inverse must contain the same number of subsets.")

    rows: list[torch.Tensor] = []
    for row_counts, row_inverse in zip(counts, inverse, strict=True):
        counts_f = row_counts.to(dtype=torch.float64)
        total = counts_f.sum()
        if bool(total <= 0):
            raise ValueError("Counts must contain at least one observation per row.")
        observed_counts = counts_f[row_inverse.to(dtype=torch.long)]
        rows.append(torch.log2(total) - torch.log2(observed_counts))

    if not rows:
        return torch.empty((0, 0), dtype=torch.float64)
    return torch.stack(rows)


class CountEntropyEstimator(ABC):
    """Interface for entropy estimators operating on exact state counts.

    Estimators may optionally define local entropy values aligned with each
    observation. Local support is explicit because a scalar entropy correction
    does not necessarily imply a unique pointwise decomposition.
    """

    name: str
    supports_local_values: bool = False

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

    def local_from_dense(self, counts: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Return local entropy values for dense counts in sample order.

        Subclasses must opt in by overriding this method and setting
        :attr:`supports_local_values` to ``True``.
        """
        raise NotImplementedError(
            f"Entropy estimator {self.name!r} does not define local values."
        )

    def local_from_sparse(
        self,
        counts: list[torch.Tensor],
        inverse: list[torch.Tensor],
    ) -> torch.Tensor:
        """Return local entropy values for sparse counts in sample order."""
        raise NotImplementedError(
            f"Entropy estimator {self.name!r} does not define local values."
        )


class PluginEstimator(CountEntropyEstimator):
    """Maximum-likelihood (plug-in) Shannon entropy estimator in bits.

    Its local entropy is the empirical surprisal
    ``-log2(p_hat(x_t))`` for each observed sample. Averaging the local values
    exactly recovers the plug-in entropy returned by :meth:`entropy_from_dense`.
    """

    name = "plugin"
    supports_local_values = True

    def entropy_from_dense(self, counts: torch.Tensor) -> torch.Tensor:
        """Estimate plug-in Shannon entropy from dense state counts."""
        counts_f, totals = _prepare_dense_counts(counts)
        return _plugin_entropy(counts_f, totals)

    def local_from_dense(self, counts: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Calculate empirical surprisal for every observation in bits."""
        counts_f, totals = _prepare_dense_counts(counts)
        return _plugin_local_from_dense(counts_f, totals, codes)

    def local_from_sparse(
        self,
        counts: list[torch.Tensor],
        inverse: list[torch.Tensor],
    ) -> torch.Tensor:
        """Calculate empirical surprisal from observed-state counts in bits."""
        return _plugin_local_from_sparse(counts, inverse)


class MillerMadowEstimator(CountEntropyEstimator):
    """Miller-Madow finite-sample correction to plug-in entropy.

    For local values, the global Miller-Madow correction for each variable set
    is distributed uniformly over its observations. This convention guarantees
    that the sample mean of local values equals the corrected entropy, but the
    additive correction is not itself a pointwise surprisal derived from a
    probability distribution.
    """

    name = "miller_madow"
    supports_local_values = True

    def entropy_from_dense(self, counts: torch.Tensor) -> torch.Tensor:
        """Estimate Miller-Madow-corrected Shannon entropy from counts."""
        counts_f, totals = _prepare_dense_counts(counts)
        plugin = _plugin_entropy(counts_f, totals)
        observed = (counts_f > 0).sum(dim=-1).to(dtype=torch.float64)
        correction = (observed - 1.0) / (2.0 * totals * _LOG_2)
        return plugin + correction

    def local_from_dense(self, counts: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Calculate uniformly corrected local entropy values in bits."""
        counts_f, totals = _prepare_dense_counts(counts)
        local = _plugin_local_from_dense(counts_f, totals, codes)
        observed = (counts_f > 0).sum(dim=-1).to(dtype=torch.float64)
        correction = (observed - 1.0) / (2.0 * totals * _LOG_2)
        return local + correction.unsqueeze(1)

    def local_from_sparse(
        self,
        counts: list[torch.Tensor],
        inverse: list[torch.Tensor],
    ) -> torch.Tensor:
        """Calculate uniformly corrected sparse local entropy values in bits."""
        local = _plugin_local_from_sparse(counts, inverse)
        corrections: list[torch.Tensor] = []
        for row_counts in counts:
            total = row_counts.sum().to(dtype=torch.float64)
            observed = torch.tensor(
                row_counts.numel(), dtype=torch.float64, device=row_counts.device
            )
            corrections.append((observed - 1.0) / (2.0 * total * _LOG_2))
        if not corrections:
            return local
        return local + torch.stack(corrections).unsqueeze(1)


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
