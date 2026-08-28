# Local information values

Local information values resolve the four dTHOI measures at the level of individual observations. They are optional because their output size grows linearly with the number of samples in addition to the number of variable sets.

## Definitions

For a variable set \(A\), observation \(t\), and an entropy estimator that defines local values, write its sample-resolved contribution as \(h_A(t)\). dTHOI requires the estimator-specific local values to satisfy

\[
\frac{1}{T}\sum_t h_A(t)=\widehat H_A.
\]

For empirical entropy this is the ordinary Shannon surprisal,

\[
h_A(t) = -\log_2 \hat p_A\!\left(x_A^{(t)}\right).
\]

For a set \(A\) of order \(k\), dTHOI defines local total correlation and local dual total correlation from the same entropy terms used by the global measures:

\[
tc_A(t) = \sum_{i\in A} h_i(t) - h_A(t),
\]

\[
dtc_A(t) = \sum_{i\in A} h_{A\setminus i}(t) - (k-1)h_A(t).
\]

Local O-information and S-information are then derived without separate estimation pipelines:

\[
o_A(t) = tc_A(t) - dtc_A(t),
\]

\[
s_A(t) = tc_A(t) + dtc_A(t).
\]

All values are reported in bits. Whenever an entropy estimator supports local values, averaging over observations exactly recovers the corresponding global TC, DTC, O-information, and S-information.

## Entropy-estimator conventions

### Empirical

The empirical estimator has the direct pointwise interpretation of Shannon surprisal shown above.

### Miller–Madow

For Miller–Madow entropy, dTHOI uses

\[
\hat H_{MM}=\hat H_{ML}+\frac{K-1}{2T\ln 2},
\]

where \(K\) is the number of observed states. When local values are requested, this scalar correction is distributed uniformly over observations:

\[
h_{MM}(t)=h_{ML}(t)+\frac{K-1}{2T\ln 2}.
\]

This convention guarantees that the sample mean of the local values equals the Miller–Madow-corrected entropy. The additive correction should not be interpreted as a pointwise surprisal derived from a corrected probability distribution.

### Schürmann

For the implemented Schürmann estimator with \(\xi=1/2\), let \(n_i\) be the count of the state containing observation \(t\), and define

\[
I(n)=\int_0^1 \frac{u^{n-1}}{1+u}\,du.
\]

The estimator is state-additive, so each observation in state \(i\) has the canonical contribution

\[
h_{Sch}(t)=\frac{\psi(T)-\psi(n_i)-(-1)^{n_i}I(n_i)}{\ln 2}.
\]

Averaging these values over the observations gives the Schürmann entropy exactly. These are finite-sample estimator contributions, not surprisals from a normalized corrected probability distribution.

### Chao–Shen

Let \(f_1\) be the number of singleton states and

\[
\widehat C=1-\frac{f_1}{T}, \qquad
\widetilde p_i=\widehat C\frac{n_i}{T}.
\]

The Chao–Shen estimator is a Horvitz–Thompson sum over observed states,

\[
\widehat H_{CS}
=-\sum_i\frac{\widetilde p_i\log_2\widetilde p_i}
{1-(1-\widetilde p_i)^T}.
\]

Dividing each state contribution equally among its \(n_i\) observations gives

\[
h_{CS}(t)
=-\frac{\widehat C\log_2\widetilde p_i}
{1-(1-\widetilde p_i)^T}.
\]

Its sample mean is exactly \(\widehat H_{CS}\). These values are Horvitz–Thompson entropy contributions rather than ordinary Shannon surprisals. If all observations are singletons then \(\widehat C=0\); both global and local Chao–Shen estimates are undefined and dTHOI returns `NaN`.

### Estimators without local values

Shrinkage, fixed Pitman–Yor, asymptotic NSB, and the coherent complete-joint estimators (`dirichlet_a1`, `dirichlet_eb`, and `nsb`) do not expose local values.

For shrinkage, unobserved nominal states receive positive probability and contribute to the estimated entropy. The surprisal of an observed sample under the shrunken probabilities therefore averages to an empirical cross-entropy, not to the shrinkage entropy; recovering the latter would require an arbitrary allocation of the unseen-state contribution.

For fixed Pitman–Yor, the posterior mean entropy contains an explicit posterior tail over unobserved symbols. There is no unique way to assign that tail entropy to the observed samples while preserving the global posterior mean.

Asymptotic NSB is a global coincidence-count estimator driven by the sample size and number of occupied states. An exact observation-level representation would amount to an arbitrary redistribution of the scalar estimate rather than a state-resolved entropy contribution.

The coherent estimators use posterior-mean entropies under one complete-joint Dirichlet model. Their entropy expectations include contributions from unobserved joint and marginal states. Assigning those contributions to observed samples would require an additional pointwise convention that is not part of the estimator.

Local support is therefore an explicit estimator capability rather than an automatic consequence of providing a scalar information estimate.

## Returned shapes

Global information measures have shape

```text
[variable_sets, datasets, 4]
```

where the final dimension is TC, DTC, O-information, and S-information.

Local values are stored separately for each dataset. Dataset \(d\) has shape

```text
[variable_sets, samples_d, 4]
```

This representation preserves datasets with unequal sample counts without padding, merging, or reshaping observations.

The public result object exposes named views:

```python
results = dthoi.information_measures(data, variable_sets, local_values=True)

results.local.total_correlation
results.local.dual_total_correlation
results.local.o_information
results.local.s_information
```

Each property is a tuple containing one tensor per dataset.

## Batch and memory behavior

Local calculations multiply the output size by the number of samples. dTHOI therefore uses two levels of bounded processing when `local_values=True`:

1. Variable-set batches are limited according to `variable sets × total samples`.
2. Singleton and leave-one-out entropy terms are processed in smaller internal chunks and accumulated directly into TC and DTC. dTHOI never needs to materialize a full `[variable_sets, interaction_order, samples]` entropy array.

The public parameter `max_local_values_per_batch` controls the target product used for this batching. A smaller value lowers peak temporary memory without changing the returned values.

For `analyze_orders`, the effective generated batch size is automatically reduced below `sets_per_batch` when necessary. Completed batches, including local arrays, are moved to CPU before the next batch is processed.

The final requested local output itself cannot be made smaller by batching. For explicit variable sets, the returned result still requires memory proportional to

```text
variable_sets × total_samples × 4 measures
```

and exhaustive local analyses scale with the full number of variable combinations.

## Internal reuse

Local values do not introduce independent TC, DTC, O-information, or S-information estimators. The implementation reuses the same state encodings, exact counts, entropy estimator, and entropy identities as the global entropy-composition path.

Repeated singleton and leave-one-out entropy terms are deduplicated within each local measure batch, so a shared lower-order variable set is estimated once and scattered to every parent set that needs it.

The persistent entropy cache remains global-only. Caching sample-resolved entropy arrays would change cache growth from approximately `O(variable sets)` to `O(variable sets × samples)`, so local arrays live only for the requesting batch.
