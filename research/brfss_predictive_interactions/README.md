# BRFSS 2023 validation case

This directory contains the historical BRFSS 2023 proof of concept used to develop and validate the predictive-interaction framework.

The **canonical framework definition now lives in `research/predictive_interactions/`**. Nothing in this directory should be treated as the required implementation of that framework.

## Why this case remains useful

BRFSS used only 12 binary predictors, so all `2^12 - 1 = 4095` non-empty subsets could be enumerated. That made it possible to validate ideas that must scale later without exhaustive enumeration:

- TC, DTC and O-information as the structural view of `P(X)`;
- matched `XGB-full` versus `XGB-additive` as the predictive view;
- `Delta_int = R2_full - R2_additive` as interaction advantage;
- same-order normalization through `Z_P` and `Z_Delta`;
- Logic Regression as a downstream semantic view;
- the empirical observation that redundancy-dominance is largely unrelated to absolute predictive power but is negatively associated with interaction advantage after controlling subset size.

The exhaustive atlas is now a **small-system benchmark**, not the intended scalable discovery algorithm.

## Dataset-specific details

Dataset: BRFSS 2023 `LLCP2023XPT.zip` / `LLCP2023.XPT`.

Outcome: `Y = _BMI5 / 100`.

Binary predictors:

- `EXERANY2`
- `CVDINFR4`
- `CVDCRHD4`
- `CVDSTRK3`
- `ASTHMA3`
- `CHCSCNC1`
- `CHCOCNC1`
- `CHCCOPD3`
- `ADDEPEV3`
- `CHCKDNY2`
- `HAVARTH4`
- `VETERAN3`

For these predictors, BRFSS `1 = Yes`, `2 = No`; Don't know, Refused and missing are treated as missing. `_LLCPWT` supplies the survey weight used in this case study.

## Prototype implementation

`prepare_brfss.py`, `xgb_atlas.py`, and `validate_xgb.py` are retained as research prototypes that reproduce the BRFSS analysis. They intentionally contain shortcuts that are **not** part of the general framework, including:

- exhaustive enumeration of all subsets;
- reduction to the 4096 possible complete Boolean states;
- BRFSS-specific preprocessing;
- one concrete XGBoost implementation and parameterization.

Future work should not generalize these implementation details. In particular, systems with many variables must operate on observed data and candidate variable sets rather than materializing a `2^p` state cube.

## Current BRFSS XGBoost result

For the full 12-variable system, the completed run gave approximately:

- `R2_XGB-additive = 0.040412`
- `R2_XGB-full = 0.040998`
- `Delta_int = 0.000586`

Across same-order subsets, O-information was essentially unrelated to size-normalized total XGBoost performance but negatively associated with size-normalized XGBoost interaction advantage. These are empirical proof-of-concept results, not framework invariants.

## Intended future use

Once dTHOI exposes the planned batched forward/backward O-information greedy search, the scalable framework should call that implementation through the contract documented in `research/predictive_interactions/contracts.py`. The BRFSS exhaustive atlas can then test where the scalable greedy paths rank against the known exhaustive same-order distributions.

The predictive evaluator is likewise replaceable: faster XGBoost implementations or other matched full/additive predictors can be plugged into the framework without changing its structural logic.
