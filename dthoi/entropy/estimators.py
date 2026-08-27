from __future__ import annotations

from abc import ABC, abstractmethod
import math
import warnings

import torch

from .counting import SparseCounts

_LOG_2 = math.log(2.0)
_EULER_GAMMA = 0.5772156649015329


def _segment_sum(
    values: torch.Tensor,
    row_ids: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Sum packed values independently for every histogram in a batch."""
    result = torch.zeros(batch_size, dtype=values.dtype, device=values.device)
    return result.scatter_add_(0, row_ids, values)


def _prepare_dense_counts(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense counts to float64 and return validated row totals."""
    if counts.ndim != 2:
        raise ValueError("Dense counts must have shape [B, Q].")
    counts_f = counts.to(dtype=torch.float64)
    totals = counts_f.sum(dim=-1)
    if bool(torch.any(totals <= 0)):
        raise ValueError("Counts must contain at least one observation per row.")
    return counts_f, totals


def _prepare_sparse_counts(
    counts: SparseCounts,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed counts to float64 and return validated row totals."""
    values = counts.values.to(dtype=torch.float64)
    totals = _segment_sum(values, counts.row_ids, counts.batch_size)
    if bool(torch.any(totals <= 0)):
        raise ValueError("Counts must contain at least one observation per row.")
    return values, totals


def _prepare_dense_local(
    counts: torch.Tensor,
    codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare dense counts and gather each observation's state count."""
    counts_f, totals = _prepare_dense_counts(counts)
    if codes.ndim != 2 or codes.shape[0] != counts_f.shape[0]:
        raise ValueError("codes must have shape [B, T] matching the count rows.")
    observed_counts = torch.gather(counts_f, 1, codes.to(dtype=torch.long))
    return counts_f, totals, observed_counts


def _prepare_sparse_local(
    counts: SparseCounts,
    inverse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare packed counts and gather each observation's state count."""
    if inverse.ndim != 2 or inverse.shape[0] != counts.batch_size:
        raise ValueError("inverse must have shape [B, T] matching the count rows.")
    values, totals = _prepare_sparse_counts(counts)
    observed_counts = values[inverse.to(dtype=torch.long)]
    return values, totals, observed_counts


def _pack_dense_counts(counts: torch.Tensor) -> SparseCounts:
    """Pack positive entries of dense counts without Python row iteration."""
    positive = counts > 0
    positions = torch.nonzero(positive, as_tuple=False)
    return SparseCounts(
        values=counts[positive],
        row_ids=positions[:, 0].to(dtype=torch.int64),
        batch_size=counts.shape[0],
    )


def _observed_states(counts: SparseCounts) -> torch.Tensor:
    """Return the number of observed states in every packed histogram."""
    return torch.bincount(counts.row_ids, minlength=counts.batch_size).to(
        dtype=torch.float64
    )


def _plugin_entropy(counts_f: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    """Compute plug-in entropy from validated float64 dense counts and totals."""
    sum_c_log2_c = torch.special.xlogy(counts_f, counts_f).sum(dim=-1) / _LOG_2
    return torch.log2(totals) - sum_c_log2_c / totals


def _plugin_entropy_from_sparse(counts: SparseCounts) -> torch.Tensor:
    """Compute plug-in entropy from packed observed-state counts."""
    values, totals = _prepare_sparse_counts(counts)
    sum_c_log2_c = _segment_sum(
        torch.special.xlogy(values, values) / _LOG_2,
        counts.row_ids,
        counts.batch_size,
    )
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
    counts: SparseCounts,
    inverse: torch.Tensor,
) -> torch.Tensor:
    """Return empirical surprisal from packed counts in original sample order."""
    _, totals, observed_counts = _prepare_sparse_local(counts, inverse)
    return torch.log2(totals).unsqueeze(1) - torch.log2(observed_counts)


def _schurmann_point_values(
    observed_counts: torch.Tensor,
    totals: torch.Tensor,
) -> torch.Tensor:
    """Return Schürmann's per-observation state contribution in bits."""
    integral = 0.5 * (
        torch.special.digamma((observed_counts + 1.0) / 2.0)
        - torch.special.digamma(observed_counts / 2.0)
    )
    signed_integral = torch.where(
        observed_counts.to(dtype=torch.int64).remainder(2) == 1,
        integral,
        -integral,
    )
    return (
        torch.special.digamma(totals)
        - torch.special.digamma(observed_counts)
        + signed_integral
    ) / _LOG_2


def _chao_shen_coverage(singletons: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    """Return Good-Turing sample coverage for each histogram."""
    return 1.0 - singletons.to(dtype=torch.float64) / totals


def _chao_shen_point_values(
    observed_counts: torch.Tensor,
    totals: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Return per-observation Chao-Shen Horvitz-Thompson contributions in bits."""
    probabilities = observed_counts / totals
    adjusted = coverage * probabilities
    detection = -torch.expm1(totals * torch.log1p(-adjusted))
    local = -coverage * torch.log(adjusted) / detection / _LOG_2
    return torch.where(
        coverage > 0.0,
        local,
        torch.full_like(local, float("nan")),
    )


class CountEntropyEstimator(ABC):
    """Interface for Shannon entropy estimators operating on exact state counts.

    Estimators receive one batch of histograms and return one entropy per batch
    row in bits. ``n_states`` is the complete state-space cardinality; for a
    binary variable set of order ``k`` it is ``2**k`` even when some states are
    unobserved.

    Estimators may optionally define local entropy values aligned with each
    observation. Local support is explicit because a scalar bias correction or
    Bayesian estimate does not necessarily imply a unique pointwise
    decomposition.
    """

    name: str
    supports_local_values: bool = False

    @abstractmethod
    def entropy_from_dense(
        self,
        counts: torch.Tensor,
        *,
        n_states: int | None = None,
    ) -> torch.Tensor:
        """Estimate one entropy per row of dense ``[B, Q]`` counts in bits."""

    @abstractmethod
    def entropy_from_sparse(
        self,
        counts: SparseCounts,
        *,
        n_states: int,
    ) -> torch.Tensor:
        """Estimate one entropy per row of packed observed-state counts in bits."""

    def local_from_dense(self, counts: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Return local entropy values for dense counts in sample order."""
        raise NotImplementedError(
            f"Entropy estimator {self.name!r} does not define local values."
        )

    def local_from_sparse(
        self,
        counts: SparseCounts,
        inverse: torch.Tensor,
    ) -> torch.Tensor:
        """Return local entropy values for packed sparse counts in sample order."""
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

    def entropy_from_dense(
        self,
        counts: torch.Tensor,
        *,
        n_states: int | None = None,
    ) -> torch.Tensor:
        """Estimate plug-in Shannon entropy from dense state counts."""
        del n_states
        counts_f, totals = _prepare_dense_counts(counts)
        return _plugin_entropy(counts_f, totals)

    def entropy_from_sparse(
        self,
        counts: SparseCounts,
        *,
        n_states: int,
    ) -> torch.Tensor:
        """Estimate plug-in Shannon entropy from packed observed-state counts."""
        del n_states
        return _plugin_entropy_from_sparse(counts)

    def local_from_dense(self, counts: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Calculate empirical surprisal for every observation in bits."""
        counts_f, totals, observed_counts = _prepare_dense_local(counts, codes)
        del counts_f
        return torch.log2(totals).unsqueeze(1) - torch.log2(observed_counts)

    def local_from_sparse(
        self,
        counts: SparseCounts,
        inverse: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate empirical surprisal from packed observed-state counts."""
        return _plugin_local_from_sparse(counts, inverse)


class MillerMadowEstimator(CountEntropyEstimator):
    """Miller-Madow finite-sample correction to plug-in entropy in bits.

    For local values, the global Miller-Madow correction for each variable set
    is distributed uniformly over its observations. This convention guarantees
    that the sample mean of local values equals the corrected entropy, but the
    additive correction is not itself a pointwise surprisal derived from a
    probability distribution.
    """

    name = "miller_madow"
    supports_local_values = True

    def entropy_from_dense(
        self,
        counts: torch.Tensor,
        *,
        n_states: int | None = None,
    ) -> torch.Tensor:
        """Estimate Miller-Madow-corrected Shannon entropy from dense counts."""
        del n_states
        counts_f, totals = _prepare_dense_counts(counts)
        plugin = _plugin_entropy(counts_f, totals)
        observed = (counts_f > 0).sum(dim=-1).to(dtype=torch.float64)
        correction = (observed - 1.0) / (2.0 * totals * _LOG_2)
        return plugin + correction

    def entropy_from_sparse(
        self,
        counts: SparseCounts,
        *,
        n_states: int,
    ) -> torch.Tensor:
        """Estimate Miller-Madow entropy from packed observed-state counts."""
        del n_states
        plugin = _plugin_entropy_from_sparse(counts)
        _, totals = _prepare_sparse_counts(counts)
        observed = _observed_states(counts).to(device=totals.device)
        correction = (observed - 1.0) / (2.0 * totals * _LOG_2)
        return plugin + correction

    def local_from_dense(self, counts: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Calculate uniformly corrected local entropy values in bits."""
        counts_f, totals, observed_counts = _prepare_dense_local(counts, codes)
        local = torch.log2(totals).unsqueeze(1) - torch.log2(observed_counts)
        observed = (counts_f > 0).sum(dim=-1).to(dtype=torch.float64)
        correction = (observed - 1.0) / (2.0 * totals * _LOG_2)
        return local + correction.unsqueeze(1)

    def local_from_sparse(
        self,
        counts: SparseCounts,
        inverse: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate uniformly corrected sparse local entropy values in bits."""
        local = _plugin_local_from_sparse(counts, inverse)
        _, totals = _prepare_sparse_counts(counts)
        observed = _observed_states(counts).to(device=totals.device)
        correction = (observed - 1.0) / (2.0 * totals * _LOG_2)
        return local + correction.unsqueeze(1)


class _PackedEstimator(CountEntropyEstimator):
    """Base for global estimators sharing one packed batched implementation."""

    def entropy_from_dense(
        self,
        counts: torch.Tensor,
        *,
        n_states: int | None = None,
    ) -> torch.Tensor:
        """Pack positive dense counts and reuse the sparse batched formula."""
        _prepare_dense_counts(counts)
        support = counts.shape[-1] if n_states is None else n_states
        return self.entropy_from_sparse(_pack_dense_counts(counts), n_states=support)


class SchurmannEstimator(_PackedEstimator):
    r"""Schürmann finite-sample entropy estimator with :math:`\xi=1/2`.

    The implemented convention is

    .. math::

        \hat H = \psi(N) - \frac{1}{N}\sum_i n_i\left[\psi(n_i)
        + (-1)^{n_i}\int_0^1 \frac{t^{n_i-1}}{1+t}\,dt\right],

    reported in bits. The integral is evaluated analytically with digamma
    functions, avoiding work proportional to each count. This is the same
    :math:`\xi=1/2` member implemented as ``entropy_2`` in ``dit``.

    For observation ``t`` in state ``i``, dTHOI uses the state-additive local
    contribution

    .. math::

        h_t = \frac{\psi(N)-\psi(n_i)-(-1)^{n_i}I(n_i)}{\ln 2},
        \qquad I(n)=\int_0^1\frac{u^{n-1}}{1+u}\,du.

    Averaging these values over observations exactly recovers ``hat H``. They
    are estimator contributions, not surprisals derived from a normalized
    corrected probability distribution.

    Assumptions
    -----------
    Samples are independent draws from a discrete distribution. Only observed
    state counts enter the estimate; the nominal state-space size is not used.
    """

    name = "schurmann"
    supports_local_values = True

    def entropy_from_sparse(
        self,
        counts: SparseCounts,
        *,
        n_states: int,
    ) -> torch.Tensor:
        """Estimate Schürmann entropy for a packed batch of state counts."""
        del n_states
        values, totals = _prepare_sparse_counts(counts)
        row_ids = counts.row_ids
        state_values = _schurmann_point_values(values, totals[row_ids])
        weights = values / totals[row_ids]
        return _segment_sum(weights * state_values, row_ids, counts.batch_size)

    def local_from_dense(self, counts: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Calculate Schürmann state-additive local contributions in bits."""
        _, totals, observed_counts = _prepare_dense_local(counts, codes)
        return _schurmann_point_values(observed_counts, totals.unsqueeze(1))

    def local_from_sparse(
        self,
        counts: SparseCounts,
        inverse: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate packed Schürmann local contributions in sample order."""
        _, totals, observed_counts = _prepare_sparse_local(counts, inverse)
        return _schurmann_point_values(observed_counts, totals.unsqueeze(1))


class ShrinkageEstimator(_PackedEstimator):
    r"""James-Stein shrinkage entropy over the full nominal binary support.

    Empirical probabilities are shrunk toward the uniform target
    :math:`t_i=1/Q`, where ``Q`` is the complete state-space cardinality. The
    shrinkage intensity is

    .. math::

        \lambda = \frac{1-\sum_i \hat p_i^2}
        {(N-1)\sum_i(t_i-\hat p_i)^2},

    clipped to ``[0, 1]``. Contributions from unobserved states are included
    analytically, so sparse execution does not materialize all ``Q`` states.

    Notes
    -----
    For a dTHOI binary variable set of order ``k``, ``Q=2**k`` is the nominal
    support. If the scientific system contains structural zeros, shrinkage
    toward a uniform distribution on all ``Q`` states is a modeling assumption.
    A sample surprisal under the shrunken probabilities would average to an
    empirical cross-entropy, not to this shrinkage entropy because unobserved
    states receive positive mass. dTHOI therefore does not expose local values
    for this estimator.
    """

    name = "shrinkage"

    def entropy_from_sparse(
        self,
        counts: SparseCounts,
        *,
        n_states: int,
    ) -> torch.Tensor:
        """Estimate shrinkage entropy for packed observed-state counts."""
        if n_states < 1:
            raise ValueError("n_states must be positive.")
        values, totals = _prepare_sparse_counts(counts)
        row_ids = counts.row_ids
        observed = _observed_states(counts).to(device=values.device)
        if bool(torch.any(observed > n_states)):
            raise ValueError("Observed states cannot exceed n_states.")

        probabilities = values / totals[row_ids]
        sum_squared = _segment_sum(
            probabilities.square(), row_ids, counts.batch_size
        )
        target = 1.0 / float(n_states)
        denominator = _segment_sum(
            (probabilities - target).square(), row_ids, counts.batch_size
        ) + (float(n_states) - observed) * target**2
        numerator = (1.0 - sum_squared) / (totals - 1.0)
        shrinkage = torch.where(
            totals <= 1.0,
            torch.ones_like(totals),
            torch.where(
                denominator > 0.0,
                numerator / denominator,
                torch.ones_like(totals),
            ),
        ).clamp_(0.0, 1.0)

        observed_probabilities = (
            shrinkage[row_ids] * target
            + (1.0 - shrinkage[row_ids]) * probabilities
        )
        observed_entropy = -_segment_sum(
            torch.special.xlogy(observed_probabilities, observed_probabilities),
            row_ids,
            counts.batch_size,
        ) / _LOG_2

        unobserved_probability = shrinkage * target
        unobserved_entropy = -(
            (float(n_states) - observed)
            * torch.special.xlogy(unobserved_probability, unobserved_probability)
            / _LOG_2
        )
        return observed_entropy + unobserved_entropy


class ChaoShenEstimator(_PackedEstimator):
    r"""Chao-Shen sample-coverage-corrected Shannon entropy in bits.

    Sample coverage is estimated as :math:`C=1-f_1/N`, where :math:`f_1` is
    the number of singleton states. Observed probabilities are rescaled by
    ``C`` and corrected by their estimated probability of detection. Writing
    :math:`\tilde p_i=Cn_i/N`, the global estimator is

    .. math::

        \hat H_{CS}=-\sum_i\frac{\tilde p_i\log_2\tilde p_i}
        {1-(1-\tilde p_i)^N}.

    Its state-additive form gives the local contribution for observation ``t``
    in state ``i``:

    .. math::

        h_t=-\frac{C\log_2\tilde p_i}{1-(1-\tilde p_i)^N}.

    Averaging over observations exactly recovers ``hat H_CS``. These values are
    Horvitz-Thompson entropy contributions, not ordinary Shannon surprisals.

    If every observation is a singleton, ``C=0`` and both global and local
    estimates are undefined; dTHOI returns ``NaN`` rather than modifying the
    singleton count with an ad-hoc numerical fallback.
    """

    name = "chao_shen"
    supports_local_values = True

    def entropy_from_sparse(
        self,
        counts: SparseCounts,
        *,
        n_states: int,
    ) -> torch.Tensor:
        """Estimate Chao-Shen entropy for packed observed-state counts."""
        del n_states
        values, totals = _prepare_sparse_counts(counts)
        row_ids = counts.row_ids
        singletons = _segment_sum(
            (values == 1.0).to(dtype=torch.float64), row_ids, counts.batch_size
        )
        coverage = _chao_shen_coverage(singletons, totals)
        state_values = _chao_shen_point_values(
            values,
            totals[row_ids],
            coverage[row_ids],
        )
        weights = values / totals[row_ids]
        entropy = _segment_sum(weights * state_values, row_ids, counts.batch_size)
        return torch.where(
            coverage > 0.0,
            entropy,
            torch.full_like(entropy, float("nan")),
        )

    def local_from_dense(self, counts: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Calculate Chao-Shen Horvitz-Thompson contributions in bits."""
        counts_f, totals, observed_counts = _prepare_dense_local(counts, codes)
        singletons = (counts_f == 1.0).sum(dim=-1)
        coverage = _chao_shen_coverage(singletons, totals)
        return _chao_shen_point_values(
            observed_counts,
            totals.unsqueeze(1),
            coverage.unsqueeze(1),
        )

    def local_from_sparse(
        self,
        counts: SparseCounts,
        inverse: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate packed Chao-Shen local contributions in sample order."""
        values, totals, observed_counts = _prepare_sparse_local(counts, inverse)
        singletons = _segment_sum(
            (values == 1.0).to(dtype=torch.float64),
            counts.row_ids,
            counts.batch_size,
        )
        coverage = _chao_shen_coverage(singletons, totals)
        return _chao_shen_point_values(
            observed_counts,
            totals.unsqueeze(1),
            coverage.unsqueeze(1),
        )


class PitmanYorEstimator(_PackedEstimator):
    r"""Posterior-mean entropy under a fixed Pitman-Yor process prior.

    Parameters
    ----------
    discount
        Pitman-Yor discount :math:`d`, restricted to ``0 <= d < 1``.
    concentration
        Concentration :math:`\alpha`. The supported countably infinite
        Pitman-Yor regime requires ``alpha >= 0`` and ``alpha > 0`` when
        ``d=0``.

    Notes
    -----
    The default ``d=1/2, alpha=0`` is the fixed convention selected for the
    initial dTHOI implementation. The returned quantity is the posterior mean
    :math:`E[H(p)\mid n,d,\alpha]`, not the entropy of posterior-mean
    probabilities. The posterior contains a Pitman-Yor tail over unobserved
    symbols, so assigning that tail entropy to observed samples has no unique
    pointwise rule. The prior also does not use the finite nominal cardinality
    ``Q=2**k``. dTHOI therefore does not expose local values for this estimator.
    """

    name = "pitman_yor"

    def __init__(self, discount: float = 0.5, concentration: float = 0.0):
        """Configure fixed Pitman-Yor hyperparameters."""
        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must satisfy 0 <= discount < 1.")
        if concentration < 0.0:
            raise ValueError("concentration must be non-negative.")
        if discount == 0.0 and concentration == 0.0:
            raise ValueError("concentration must be positive when discount is zero.")
        self.discount = float(discount)
        self.concentration = float(concentration)

    def entropy_from_sparse(
        self,
        counts: SparseCounts,
        *,
        n_states: int,
    ) -> torch.Tensor:
        """Estimate fixed Pitman-Yor posterior-mean entropy in bits."""
        del n_states
        values, totals = _prepare_sparse_counts(counts)
        row_ids = counts.row_ids
        observed = _observed_states(counts).to(device=values.device)
        d = self.discount
        alpha = self.concentration

        observed_mass = totals - observed * d
        head_entropy = torch.special.digamma(observed_mass + 1.0) - _segment_sum(
            ((values - d) / observed_mass[row_ids])
            * torch.special.digamma(values - d + 1.0),
            row_ids,
            counts.batch_size,
        )

        tail_mass = alpha + observed * d
        one_minus_d = torch.tensor(
            1.0 - d, dtype=torch.float64, device=values.device
        )
        tail_entropy = torch.special.digamma(tail_mass + 1.0) - torch.special.digamma(
            one_minus_d
        )
        tail_weight = tail_mass / (alpha + totals)
        split_entropy = (
            torch.special.digamma(alpha + totals + 1.0)
            - tail_weight * torch.special.digamma(tail_mass + 1.0)
            - (1.0 - tail_weight) * torch.special.digamma(observed_mass + 1.0)
        )
        entropy_nats = (
            (1.0 - tail_weight) * head_entropy
            + tail_weight * tail_entropy
            + split_entropy
        )
        return entropy_nats / _LOG_2


class AsymptoticNsbEstimator(_PackedEstimator):
    r"""Few-coincidence asymptotic NSB entropy estimator.

    The estimate is

    .. math::

        \hat H = \gamma - \ln 2 + 2\ln N - \psi(\Delta),
        \qquad \Delta=N-K_1,

    in nats before conversion to bits, where ``K1`` is the number of observed
    states and ``Delta`` is the number of coincidences. The asymptotic regime is
    ``N/Q -> 0``; dTHOI uses the known nominal binary cardinality ``Q=2**k`` to
    diagnose this condition separately from ``K1``.

    Parameters
    ----------
    max_sample_to_state_ratio
        Upper ``N/Q`` ratio used only to warn about use outside severe
        undersampling. The estimate is still returned. With no coincidences
        (``Delta=0``), the estimator is undefined and returns ``NaN``.

    Notes
    -----
    ANSB is a global coincidence-count estimator depending on ``N`` and the
    number of occupied states rather than on state-specific entropy terms. Any
    exact assignment to individual observations would therefore require an
    arbitrary redistribution of the scalar estimate. dTHOI does not expose
    local values. Its asymptotic assumptions also generally fail for low-order
    binary marginals used in multivariate information measures.
    """

    name = "ansb"

    def __init__(self, max_sample_to_state_ratio: float = 0.1):
        """Configure the diagnostic threshold for the asymptotic regime."""
        if max_sample_to_state_ratio <= 0.0:
            raise ValueError("max_sample_to_state_ratio must be positive.")
        self.max_sample_to_state_ratio = float(max_sample_to_state_ratio)

    def entropy_from_sparse(
        self,
        counts: SparseCounts,
        *,
        n_states: int,
    ) -> torch.Tensor:
        """Estimate asymptotic NSB entropy for packed observed-state counts."""
        if n_states < 1:
            raise ValueError("n_states must be positive.")
        values, totals = _prepare_sparse_counts(counts)
        observed = _observed_states(counts).to(device=values.device)
        ratios = totals / float(n_states)
        if bool(torch.any(ratios > self.max_sample_to_state_ratio)):
            warnings.warn(
                "Asymptotic NSB is intended for severe undersampling (N/Q -> 0); "
                "at least one histogram exceeds max_sample_to_state_ratio.",
                RuntimeWarning,
                stacklevel=3,
            )

        coincidences = totals - observed
        entropy_nats = (
            _EULER_GAMMA
            - _LOG_2
            + 2.0 * torch.log(totals)
            - torch.special.digamma(coincidences)
        )
        entropy = entropy_nats / _LOG_2
        return torch.where(
            coincidences > 0.0,
            entropy,
            torch.full_like(entropy, float("nan")),
        )


def resolve_estimator(estimator: str | CountEntropyEstimator) -> CountEntropyEstimator:
    """Resolve a registered entropy estimator name or estimator instance.

    Parameters
    ----------
    estimator
        Registered estimator name or an existing :class:`CountEntropyEstimator`.
        Scientific names are ``"empirical"``, ``"miller_madow"``,
        ``"schurmann"``, ``"shrinkage"``, ``"chao_shen"``, ``"pitman_yor"``,
        and ``"ansb"``. ``"pitman_yor"`` uses fixed ``d=1/2, alpha=0``.

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
    if key in {"schurmann", "schuermann"}:
        return SchurmannEstimator()
    if key in {"shrinkage", "shrink"}:
        return ShrinkageEstimator()
    if key in {"chao_shen", "chaoshen"}:
        return ChaoShenEstimator()
    if key in {"pitman_yor", "py"}:
        return PitmanYorEstimator()
    if key in {"ansb", "asymptotic_nsb"}:
        return AsymptoticNsbEstimator()
    raise ValueError(f"Unknown estimator: {estimator!r}.")