# Local information values

Local information values resolve the four dTHOI measures at the level of individual observations. They are optional because their output size grows linearly with the number of samples in addition to the number of variable sets.

## Definitions

For a variable set \(A\), observation \(t\), and empirical probability \(\hat p_A\), the local Shannon entropy value is the surprisal

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

All values are reported in bits.

For empirical entropy, averaging over observations exactly recovers the corresponding global measure:

\[
\frac{1}{T}\sum_t tc_A(t)=TC_A,
\]

and analogously for DTC, O-information, and S-information.

## Entropy-estimator convention

The empirical estimator has a direct pointwise interpretation as Shannon surprisal.

For Miller–Madow entropy, dTHOI uses

\[
\hat H_{MM}=\hat H_{ML}+\frac{K-1}{2T\ln 2},
\]

where \(K\) is the number of observed states. When local values are requested, this scalar correction is distributed uniformly over observations:

\[
h_{MM}(t)=h_{ML}(t)+\frac{K-1}{2T\ln 2}.
\]

This convention guarantees that the sample mean of the local values equals the Miller–Madow-corrected entropy. The additive correction should not be interpreted as a pointwise surprisal derived from a corrected probability distribution.

Future entropy estimators are not required to define local values. Local support is an explicit estimator capability rather than an automatic consequence of providing a scalar entropy estimate.

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

Local values do not introduce independent TC, DTC, O-information, or S-information estimators. The implementation reuses the same state encodings, exact counts, entropy estimator, and entropy identities as the global path.

Repeated singleton and leave-one-out entropy terms are deduplicated within each local measure batch, so a shared lower-order variable set is estimated once and scattered to every parent set that needs it.

The persistent entropy cache remains global-only. Caching sample-resolved entropy arrays would change cache growth from approximately `O(variable sets)` to `O(variable sets × samples)`, so local arrays live only for the requesting batch.
