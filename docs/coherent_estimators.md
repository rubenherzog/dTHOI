# Coherent information estimators

`dthoi.information_measures(..., estimator=...)` supports three coherent Bayesian estimators in addition to the entropy-composition estimators used elsewhere in dTHOI:

- `dirichlet_a1`
- `dirichlet_eb`
- `nsb`

They use the same public API and return the same `InformationMeasures` object. Their internal statistical contract is different: one complete-joint posterior is fitted for each requested variable set, and every entropy needed for TC, DTC, O-information, and S-information is evaluated as a marginal of that same posterior.

## Model and units

For a binary variable set of order `k`, let `p_x` denote the probability of state `x`. Conditional on total concentration `A`,

\[
P=(p_x)_{x\in\{0,1\}^k}\sim\operatorname{Dirichlet}(A/2^k,\ldots,A/2^k).
\]

Marginalization preserves total concentration `A`. For a marginal containing `r` variables, let `n_j` be its positive observed counts, `N` the number of observations, and

\[
\alpha=A/2^r,\qquad \beta_x=n_x+\alpha.
\]

The posterior-mean Shannon entropy is

\[
E[H\mid D,A]
=\frac{1}{\ln 2}\left[
\psi(N+A+1)
-\frac{1}{N+A}\sum_x\beta_x\psi(\beta_x+1)
\right].
\]

Unobserved states have `n_x=0` and are aggregated analytically; they are never enumerated. Entropy and all derived measures are reported in bits.

The dTHOI definitions remain unchanged:

\[
TC=\sum_i H(X_i)-H(X_1,\ldots,X_k),
\]

\[
DTC=\sum_i H(X_{-i})-(k-1)H(X_1,\ldots,X_k),
\]

\[
O=TC-DTC,\qquad S=TC+DTC.
\]

Because TC and DTC are non-negative for every distribution, their posterior expectations under one coherent model are also non-negative apart from floating-point roundoff.

## Estimator choices

`dirichlet_a1` fixes `A=1`.

`dirichlet_eb` chooses `A` by maximizing the complete-joint Dirichlet-multinomial evidence. Terms independent of `A` are omitted:

\[
\log p(n\mid A)
=\log\Gamma(A)-\log\Gamma(N+A)
+\sum_{x\in\mathrm{obs}}
\left[\log\Gamma(n_x+A/2^k)-\log\Gamma(A/2^k)\right].
\]

`nsb` integrates over the same `A` using the entropy-flat NSB hyperprior. For complete-joint support size `Q=2^k`,

\[
\xi(A)=\frac{\psi(A+1)-\psi(A/Q+1)}{\ln 2},
\]

and the hyperprior is proportional to

\[
\frac{d\xi}{dA}
=\frac{\psi_1(A+1)-Q^{-1}\psi_1(A/Q+1)}{\ln 2}.
\]

EB and NSB use a deterministic 72-point grid equally spaced in `log(A)`, from `1e-4` to `max(1e4, 100*N)`. NSB integration includes the `A` Jacobian. The grid depends only on the dataset sample count and interaction order, so a variable set returns the same value whether it is evaluated alone or inside a larger execution batch. If the EB optimum or non-negligible NSB posterior mass remains at a grid boundary, dTHOI emits a `RuntimeWarning` rather than silently changing the estimator as a function of batch composition.

## Common execution path

```python
result = dthoi.information_measures(
    data,
    variable_sets,
    estimator="nsb",
)
```

Internally, dTHOI resolves the selected estimator into either an entropy-composition provider or a coherent complete-joint provider. `information_measures` and `analyze_orders` use the same dispatcher and batching machinery.

For coherent estimators with interaction order up to 63, dTHOI reuses its exact batched binary state encoders and packed sparse observed-state counts. Within each parent-set batch:

1. repeated complete variable sets are deduplicated;
2. complete-joint counts are calculated once;
3. shared singleton variable sets are deduplicated and counted once;
4. shared leave-one-out variable sets are deduplicated and counted once;
5. posterior entropy grids are evaluated in Torch batches;
6. TC and DTC are combined with the same common definitions used by the entropy path.

The same lower-order count table can therefore be reused by multiple parent sets even when its posterior entropy differs because those parents select different values of `A`.

For interaction order greater than 63, dTHOI uses an exact observed-row fallback. This path avoids machine-word packing while preserving the same posterior equations.

## Why these estimators are not standalone entropy estimators

`estimate_entropy` remains limited to estimators whose result is determined by the requested variable set alone. For coherent EB and NSB, the concentration is selected from the complete parent joint distribution. The entropy assigned to a leave-one-out marginal therefore depends on which complete parent set is being analyzed.

Registering these methods as ordinary standalone entropy estimators would allow the joint and its marginals to select different posterior models and would destroy the coherence the method is intended to provide. This distinction stays internal to the information-measure API: users still select the method with the same `estimator=` parameter.

## Assumptions and limitations

The current implementation assumes exact binary observations and independent draws from one stationary discrete distribution per dataset. Multiple datasets remain separate and are estimated independently. Structural zeros are not encoded explicitly; every nominal binary state receives symmetric prior mass.

The nominal `2**k` state space is never allocated. Computational feasibility therefore does not imply statistical identifiability. If complete-state repetitions are scarce, many high-order distributions remain compatible with the observations and the posterior estimate can be strongly prior-dependent.

Coherent posterior-mean estimators do not expose local sample-resolved values. Their entropy expectations contain contributions from unobserved states, and assigning those contributions to observed samples would require an additional convention that is not part of the estimator.
