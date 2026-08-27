# dTHOI

dTHOI is a Torch-based library for higher-order information analysis of discrete multivariate data. It is designed for scientific users who want to study how information is shared across combinations of variables without having to manage the underlying counting and memory strategy.

The current core supports binary observations, Shannon entropy estimation, total correlation, dual total correlation, O-information, S-information, and analysis across multiple interaction orders.

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

By default, dTHOI uses the empirical entropy estimator. A Miller–Madow correction is also available:

```python
h = dthoi.estimate_entropy(
    data,
    variable_sets,
    entropy_estimator="miller_madow",
)
```

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

## Public API

The main user-facing functions are:

- `prepare_data`: prepare one or more datasets for repeated analyses.
- `estimate_entropy`: estimate Shannon entropy for selected variable sets.
- `information_measures`: calculate TC, DTC, O-information, and S-information.
- `analyze_orders`: evaluate all variable combinations across a range of interaction orders.

Implementation details such as state counting, internal entropy reuse, and memory strategy are handled automatically.
