# dTHOI

dTHOI is a Torch-based library for higher-order information analysis of discrete multivariate data. It is designed for scientific users who want to study how information is shared across combinations of variables without having to manage the underlying counting and memory strategy.

The current core supports binary observations, multiple information estimators, total correlation, dual total correlation, O-information, S-information, sample-resolved local values, and analysis across multiple interaction orders.

## Basic use

Observations are arranged as samples × variables. Variable sets are specified by their column indices.

```python
import torch
import dthoi

data = torch.tensor(
    [[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
    dtype=torch.uint8,
)
variable_sets = [[0, 1, 2], [1, 2, 3]]
results = dthoi.information_measures(data, variable_sets)

results.total_correlation
results.dual_total_correlation
results.o_information
results.s_information
```

The returned measures are Torch tensors with one row per variable set and one column per dataset.

## Choosing an information estimator

All information estimators are selected through the same public API:

```python
results = dthoi.information_measures(
    data,
    variable_sets,
    estimator="nsb",
)
```

Entropy-composition estimators are:

- `"empirical"`: maximum-likelihood entropy from observed frequencies.
- `"miller_madow"`: first-order Miller–Madow finite-sample correction.
- `"schurmann"`: Schürmann finite-sample estimator with `xi=1/2`.
- `"shrinkage"`: James–Stein shrinkage toward the uniform binary distribution.
- `"chao_shen"`: Chao–Shen sample-coverage correction.
- `"pitman_yor"`: posterior-mean entropy for the fixed Pitman–Yor convention `d=1/2, alpha=0`.
- `"ansb"`: asymptotic NSB entropy estimator for the few-coincidence regime.

Coherent complete-joint estimators are:

- `"dirichlet_a1"`: one symmetric Dirichlet posterior with total concentration `A=1`.
- `"dirichlet_eb"`: the same model with `A` selected by complete-joint Dirichlet-multinomial evidence.
- `"nsb"`: an NSB entropy-flat mixture over the same complete-joint concentration `A`.

The two families have different internal statistical contracts but the same scientific API. Entropy-composition methods estimate reusable subset entropies and combine them into TC, DTC, O-information, and S-information. Coherent estimators fit one model to each complete variable set and derive every singleton and leave-one-out entropy from that same posterior. See `docs/coherent_estimators.md` for equations and assumptions.

The former `entropy_estimator=` argument to `information_measures` and `analyze_orders` remains accepted as a compatibility alias. New code should use `estimator=`.

## Entropy

Standalone Shannon entropy estimation remains available separately:

```python
h = dthoi.estimate_entropy(
    data,
    variable_sets,
    entropy_estimator="shrinkage",
)
```

`estimate_entropy` accepts the entropy-composition estimators listed above. Complete-joint estimators such as `"nsb"` are intentionally not exposed here because their marginal entropy estimates depend on the parent joint variable set.

All entropy estimates are reported in bits. Shrinkage treats `2**order` as the nominal binary support, which is a modeling assumption when structural zeros are known. Chao–Shen returns `NaN` when every observation is a singleton. The fixed Pitman–Yor estimator uses a countably infinite prior. ANSB uses the nominal `2**order` support to diagnose its severe-undersampling regime.

## Coherent estimator scaling

The coherent estimators never allocate the nominal `2**order` state space. For interaction orders up to 63 they reuse dTHOI's batched exact state encoding and sparse observed-state counting. Shared singleton and leave-one-out variable sets are deduplicated within each parent batch before their entropy grids are evaluated. Wider interaction sets use an exact observed-row fallback.

This removes exponential state-space allocation, not the statistical identifiability problem. When almost every observed complete state is unique, high-order estimates can remain strongly prior-dependent even though the computation itself is feasible.

## Local information values

Sample-resolved TC, DTC, O-information, and S-information can be requested explicitly:

```python
results = dthoi.information_measures(
    data,
    variable_sets,
    estimator="empirical",
    local_values=True,
)
local_o = results.local.o_information[0]
```

Local values are defined for `"empirical"`, `"miller_madow"`, `"schurmann"`, and `"chao_shen"`. Other estimators contain global or unobserved-state contributions without a unique observation-level allocation and raise `NotImplementedError` when `local_values=True`. See `docs/local_values.md` for equations and conventions.

## Reusing prepared data

```python
prepared = dthoi.prepare_data(data)
h = dthoi.estimate_entropy(prepared, variable_sets)
results = dthoi.information_measures(prepared, variable_sets, estimator="nsb")
```

## Exploring interaction orders

`analyze_orders` uses the same estimator dispatcher as `information_measures`:

```python
results = dthoi.analyze_orders(
    data,
    min_order=3,
    max_order=4,
    estimator="dirichlet_eb",
)
```

Results are returned in bounded pieces so large combination spaces do not need to be evaluated as one monolithic object. Estimators may reduce their internal working batch below `sets_per_batch` to control temporary memory without changing the requested variable sets.

ANSB deserves an additional caveat for higher-order measures: its severe-undersampling assumption must hold for every entropy term used to build TC, DTC, O-information, and S-information. It normally fails for binary singleton marginals (`Q=2`), so ANSB is primarily an estimator for direct high-order entropy unless those constituent assumptions are otherwise justified.

## Public API

- `prepare_data`: prepare one or more datasets for repeated analyses.
- `estimate_entropy`: estimate standalone Shannon entropy for selected variable sets.
- `information_measures`: calculate TC, DTC, O-information, and S-information with one selected estimator.
- `analyze_orders`: evaluate variable combinations across interaction orders using the same estimator choices.

Implementation details such as estimator family, state counting, internal entropy reuse, posterior-grid integration, and memory strategy are handled automatically.
