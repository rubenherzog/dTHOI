# dTHOI

dTHOI is a Torch-based library for higher-order information analysis of discrete multivariate data. It is designed for scientific users who want to study how information is shared across combinations of variables without having to manage the underlying counting and memory strategy.

The current core supports binary observations, multiple Shannon entropy estimators, total correlation, dual total correlation, O-information, S-information, sample-resolved local values, and analysis across multiple interaction orders.

## Basic use

Observations are arranged as samples × variables. Variable sets are specified by their column indices.

```python
import torch
import dthoi

data = torch.tensor(
    [
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
    ],
    dtype=torch.uint8,
)

variable_sets = [
    [0, 1, 2],
    [1, 2, 3],
]

results = dthoi.information_measures(data, variable_sets)

results.total_correlation
results.dual_total_correlation
results.o_information
results.s_information
```

The returned measures are Torch tensors with one row per variable set and one column per dataset.

## Entropy

```python
h = dthoi.estimate_entropy(data, variable_sets)
```

By default, dTHOI uses empirical entropy. The same batched API supports:

- `"empirical"`: maximum-likelihood entropy from observed frequencies.
- `"miller_madow"`: first-order Miller–Madow finite-sample correction.
- `"schurmann"`: Schürmann finite-sample estimator with `xi=1/2`.
- `"shrinkage"`: James–Stein shrinkage toward the uniform distribution over the full binary state space.
- `"chao_shen"`: Chao–Shen sample-coverage correction for incomplete sampling.
- `"pitman_yor"`: posterior-mean entropy for a fixed Pitman–Yor prior with `d=1/2` and `alpha=0`.
- `"ansb"`: asymptotic NSB estimator for severe undersampling with a small number of coincidences.

For example:

```python
h = dthoi.estimate_entropy(
    data,
    variable_sets,
    entropy_estimator="shrinkage",
)
```

All entropy estimates are reported in bits. Shrinkage treats `2**order` as the nominal binary support, which is a modeling assumption when structural zeros are known. Chao–Shen returns `NaN` when every observation is a singleton. The fixed Pitman–Yor estimator uses a countably infinite prior and therefore does not impose the finite binary support. ANSB uses the nominal `2**order` support only to diagnose its `N/Q -> 0` regime and returns `NaN` when there are no repeated observations.

## Local information values

Sample-resolved TC, DTC, O-information, and S-information can be requested explicitly. They are not computed by default because their output size scales with the number of observations.

```python
results = dthoi.information_measures(
    data,
    variable_sets,
    local_values=True,
)

local_o = results.local.o_information[0]
```

Each item in `results.local.o_information` corresponds to one dataset and has shape `variable_sets × samples`. Datasets with different sample counts remain separate.

Local calculations are automatically subdivided according to the product of variable sets and samples. The default memory-control target can be reduced for large datasets:

```python
results = dthoi.information_measures(
    data,
    variable_sets,
    local_values=True,
    max_local_values_per_batch=250_000,
)
```

Local values are supported for `"empirical"`, `"miller_madow"`, `"schurmann"`, and `"chao_shen"`. Empirical values are Shannon surprisals. Miller–Madow distributes its scalar correction uniformly over observations. Schürmann uses its canonical state-additive finite-sample contribution, and Chao–Shen uses the observation-level contribution induced by its Horvitz–Thompson state sum. In every supported case, averaging local entropy values reproduces the corresponding global entropy exactly, so the same property holds for local TC, DTC, O-information, and S-information.

Shrinkage, fixed Pitman–Yor, and ANSB do not expose local values. Shrinkage and Pitman–Yor contain entropy contributions from unobserved states, while ANSB is a global coincidence-count functional; assigning these contributions to the observed samples would require an arbitrary convention. See `docs/local_values.md` for equations and details.

## Reusing prepared data

For repeated analyses of the same observations, the data can be prepared once:

```python
prepared = dthoi.prepare_data(data)

h = dthoi.estimate_entropy(prepared, variable_sets)
results = dthoi.information_measures(prepared, variable_sets)
```

## Exploring interaction orders

```python
results = dthoi.analyze_orders(
    data,
    min_order=3,
    max_order=4,
)

for result in results:
    print(result.order)
    print(result.variable_sets)
    print(result.information.o_information)
```

Results are returned in pieces so large combination spaces do not need to be evaluated as one monolithic object. Most users can ignore this detail; `sets_per_batch` can be reduced only when lower peak memory use is needed.

Local values can also be requested during an exhaustive order analysis. When enabled, dTHOI automatically reduces the effective number of variable sets per batch if the sample count makes the requested `sets_per_batch` too large. Completed local batches are moved to CPU before the next generated batch is evaluated. The total returned result can nevertheless be large because exhaustive local output scales as `variable sets × samples`.

ANSB deserves an additional caveat for higher-order measures: its severe-undersampling assumption must hold for every entropy term used to build TC, DTC, O-information, and S-information. It normally fails for binary singleton marginals (`Q=2`), so ANSB is primarily an estimator for direct high-order entropy unless those constituent assumptions are otherwise justified.

## Public API

The main user-facing functions are:

- `prepare_data`: prepare one or more datasets for repeated analyses.
- `estimate_entropy`: estimate Shannon entropy for selected variable sets.
- `information_measures`: calculate TC, DTC, O-information, and S-information, optionally per observation.
- `analyze_orders`: evaluate all variable combinations across a range of interaction orders.

Implementation details such as state counting, internal entropy reuse, and memory strategy are handled automatically.