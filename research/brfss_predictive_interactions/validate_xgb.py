from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import xgb_atlas as xa


def main() -> None:
    workdir = Path(os.environ.get("DTHOI_POC_WORKDIR", Path(__file__).resolve().parent / "work")).resolve()
    outdir = workdir / "framework"
    atlas = pd.read_csv(outdir / "xgb_atlas.csv")
    frame = pd.read_pickle(workdir / "selected.pkl")

    xraw = frame[xa.XCOLS].to_numpy(float)
    x = np.full_like(xraw, np.nan)
    x[xraw == 1] = 1.0
    x[xraw == 2] = 0.0
    y = frame[xa.YCOL].to_numpy(float) / 100.0
    w = frame[xa.WCOL].to_numpy(float)
    y[(~np.isfinite(y)) | (y < 12) | (y >= 100)] = np.nan
    w[(~np.isfinite(w)) | (w <= 0)] = np.nan

    xok = np.all(np.isfinite(x), axis=1) & np.isfinite(w)
    x_idx = np.flatnonzero(xok)
    pred_idx = x_idx[np.isfinite(y[x_idx])]
    fold_all = xa.make_fold_ids(len(x_idx), 5, xa.SEED + 100)
    fold_by_row = np.full(len(frame), -1, dtype=np.int16)
    fold_by_row[x_idx] = fold_all
    folds = fold_by_row[pred_idx]
    codes = xa.encode_rows(x[pred_idx])
    yy, ww = y[pred_idx], w[pred_idx]

    train = folds != 0
    stats = xa._full_state_stats(codes[train], yy[train], ww[train])
    sw, swy = xa._aggregate_subset(stats[0], stats[1], xa.NST - 1)
    common = dict(
        num_boost_round=100,
        max_depth=4,
        learning_rate=0.2,
        reg_lambda=5.0,
        min_child_weight=1.0,
        seed=xa.SEED + 5000,
    )
    pred_add = xa._fit_predict_xgb(sw, swy, xa.N, additive=True, **common)
    pred_full = xa._fit_predict_xgb(sw, swy, xa.N, additive=False, **common)

    max_add_mixed = 0.0
    max_full_mixed = 0.0
    for i in range(xa.N):
        for j in range(i + 1, xa.N):
            bases = np.array([
                state for state in range(xa.NST)
                if (state & (1 << i)) == 0 and (state & (1 << j)) == 0
            ])
            add_diff = (
                pred_add[bases]
                + pred_add[bases | (1 << i) | (1 << j)]
                - pred_add[bases | (1 << i)]
                - pred_add[bases | (1 << j)]
            )
            full_diff = (
                pred_full[bases]
                + pred_full[bases | (1 << i) | (1 << j)]
                - pred_full[bases | (1 << i)]
                - pred_full[bases | (1 << j)]
            )
            max_add_mixed = max(max_add_mixed, float(np.max(np.abs(add_diff))))
            max_full_mixed = max(max_full_mixed, float(np.max(np.abs(full_diff))))

    singleton_delta = float(atlas.loc[atlas["order"] == 1, "delta_int_xgb"].abs().max())
    if singleton_delta > 1e-12:
        raise AssertionError(f"Singleton XGB interaction gain is nonzero: {singleton_delta}")
    if max_add_mixed > 1e-4:
        raise AssertionError(f"XGB-additive has a mixed second difference {max_add_mixed}")

    full12 = atlas.loc[atlas["order"] == 12].iloc[0]
    summary = {
        "singleton_max_abs_delta_int": singleton_delta,
        "additive_constraint_max_abs_mixed_second_difference": max_add_mixed,
        "full_model_max_abs_mixed_second_difference": max_full_mixed,
        "full12_R2_xgb_additive": float(full12["R2_xgb_additive_crossfit"]),
        "full12_R2_xgb_full": float(full12["R2_xgb_full_crossfit"]),
        "full12_delta_int": float(full12["delta_int_xgb"]),
    }
    (outdir / "xgb_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
