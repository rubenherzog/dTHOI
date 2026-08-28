# Predictive interaction discovery framework

Status: **conceptual research framework**. This directory defines the scientific workflow and integration contracts. It is not part of the public `dthoi` API and it intentionally does not freeze the implementation of greedy search, XGBoost, or semantic models.

## Goal

The framework connects three distinct views of a discrete multivariate system without forcing one to determine the others:

1. **Predictor structure** from `P(X)`: TC, DTC and O-information in bits.
2. **Predictive structure** from `P(Y | X_S)`: total out-of-sample predictive power and the gain attributable to allowing interactions.
3. **Semantics**: interpretable logical structure extracted only after predictive discovery.

The main scientific question is not whether O-information is a predictor of `Y`, but whether the structural organization of a variable set is related to the amount of predictive non-additivity that the set can express.

## Scalable default

Exhaustive subset enumeration is **not** the default algorithm. It is reserved for small-system validation.

For a system with `p` variables, choose a scientifically relevant maximum interaction order `K << p`. For every outer training fold:

1. Draw a same-order random reference sample `R_k` for each `k = 3, ..., K` using `X` only.
2. Generate O-information-extreme candidates using two simple structural trajectories:
   - forward greedy toward low O-information;
   - forward greedy toward high O-information.
3. From each forward endpoint at order `K`, trace a backward greedy path by removing one variable at a time while preserving the corresponding O-information extreme.
4. Deduplicate the union of forward and backward candidates. Candidate generation never sees `Y`.
5. Evaluate the candidate sets and the random reference sets with the same out-of-sample predictive folds.
6. Standardize structural and predictive quantities against the random same-order reference distribution.
7. Apply semantic interpretation only to predictive candidates that warrant interpretation; semantics does not guide structural or predictive discovery.

Forward and backward paths are deliberately complementary rather than separate optimizers. Their overlap is a simple diagnostic of path dependence and structural stability.

## Canonical quantities

For a variable set `S` of order `k`:

- `Omega(S)`: O-information of `X_S`, in bits.
- `P(S)`: out-of-sample predictive score of the flexible model. For continuous outcomes the current proof of concept uses `R^2`.
- `Delta_int(S) = P_full(S) - P_additive(S)`: predictive gain from allowing interactions while otherwise matching the predictive models as closely as possible.

Using the random reference sets of the same order:

- `Z_Omega = (Omega - mean_k(Omega)) / sd_k(Omega)`
- `Z_P = (P - mean_k(P)) / sd_k(P)`
- `Z_Delta = (Delta_int - mean_k(Delta_int)) / sd_k(Delta_int)`

These Z scores are **same-order discovery scores**, not p-values. A later inferential layer may add permutation/bootstrap calibration, but the framework does not require a specific inferential implementation.

## Candidate sources

Every evaluated set carries a source label. The canonical labels are:

- `reference_random`
- `forward_low_omega`
- `forward_high_omega`
- `backward_low_omega`
- `backward_high_omega`

A set reached by more than one path is stored once with all applicable provenance labels.

## Validation contract

All procedures that use `Y` are evaluated out of sample. Candidate generation is X-only, but for evaluation of the complete discovery procedure it should run independently inside each outer training fold. This also provides a direct measure of candidate/path stability across folds.

The primary relationships are assessed after controlling subset order, preferably through `Z_Omega`, `Z_P`, and `Z_Delta`. Raw pooled relationships across different orders are descriptive only.

## Extension points

The framework deliberately depends on contracts rather than fixed implementations.

### O-information candidate generator

**Future dTHOI integration point.** The intended implementation is a batched, Torch-first forward/backward greedy search provided by dTHOI. It should consume data plus candidate variable sets and return O-information values without materializing combinatorial spaces. This research branch must not duplicate that future numerical engine.

Until that API exists, experimental code may provide an adapter with the same conceptual contract. Any such adapter is provisional.

### Predictive evaluator

The current BRFSS proof of concept uses matched `XGB-full` and `XGB-additive` models, but the framework does not prescribe the implementation, training shortcut, hyperparameters, or even XGBoost itself. A predictive evaluator must only provide matched full/additive out-of-sample scores on the supplied variable sets and folds.

Fast XGBoost implementations can replace the current prototype without changing any framework logic.

### Semantic interpreter

Logic Regression is the current proof-of-concept interpreter. It is optional and downstream. Future semantic models may replace it as long as they operate on already-discovered candidates and expose stability plus an interpretable representation.

## Canonical output

The core artifact is a long table with one row per evaluated set per evaluation context. At minimum it contains:

`variables, order, source, fold, Omega_bits, Z_Omega, P_full, P_additive, Delta_int, Z_P, Z_Delta`

Optional semantic columns are appended later rather than mixed into candidate generation.

## What is intentionally not fixed here

- the internal dTHOI greedy implementation;
- entropy batching/caching details;
- XGBoost implementation or hyperparameters;
- the number of random reference sets;
- Logic Regression implementation;
- inferential calibration beyond the same-order reference normalization;
- dataset-specific preprocessing.

Those are replaceable components. The invariant is the scientific separation:

`X-only structural candidates -> matched out-of-sample prediction -> same-order normalization -> optional semantics`.

## Small-system exhaustive mode

When exhaustive enumeration is feasible, it is used only as a validation benchmark: quantify where the scalable greedy candidates rank among all same-order sets and verify that the scalable pipeline reproduces the same structural/predictive quantities for identical sets. The BRFSS 12-variable experiment is retained for exactly this purpose.
