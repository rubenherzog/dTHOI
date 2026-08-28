from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import pearsonr, spearmanr

XCOLS = [
    "EXERANY2","CVDINFR4","CVDCRHD4","CVDSTRK3","ASTHMA3","CHCSCNC1",
    "CHCOCNC1","CHCCOPD3","ADDEPEV3","CHCKDNY2","HAVARTH4","VETERAN3"
]
YCOL = "_BMI5"
WCOL = "_LLCPWT"
N = len(XCOLS)
NST = 1 << N
SEED = 20230827
POWERS = 1 << np.arange(N, dtype=np.int64)
STATE_IDS = np.arange(NST, dtype=np.int64)
BITS = [None] * NST
PROJ = [None] * NST
LOCAL_STATES = {k: ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(float) for k in range(1, N + 1)}
for mask in range(1, NST):
    bits = [j for j in range(N) if mask & (1 << j)]
    BITS[mask] = bits
    proj = np.zeros(NST, dtype=np.int64)
    for local_bit, j in enumerate(bits):
        proj |= ((STATE_IDS >> j) & 1) << local_bit
    PROJ[mask] = proj

def encode_rows(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.int64) * POWERS).sum(axis=1)

def make_fold_ids(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    f = np.empty(n, dtype=np.int16)
    f[p] = np.arange(n) % k
    return f

def weighted_r2_tss(y: np.ndarray, w: np.ndarray) -> float:
    ym = float(np.average(y, weights=w))
    return float(np.sum(w * (y - ym) ** 2))


def _full_state_stats(codes: np.ndarray, y: np.ndarray, w: np.ndarray):
    return (
        np.bincount(codes, weights=w, minlength=NST).astype(float),
        np.bincount(codes, weights=w * y, minlength=NST).astype(float),
        np.bincount(codes, weights=w * y * y, minlength=NST).astype(float),
    )

def _aggregate_subset(full_sw: np.ndarray, full_swy: np.ndarray, mask: int):
    proj = PROJ[mask]
    k = len(BITS[mask])
    nsub = 1 << k
    sw = np.bincount(proj, weights=full_sw, minlength=nsub).astype(float)
    swy = np.bincount(proj, weights=full_swy, minlength=nsub).astype(float)
    return sw, swy


def _fit_predict_xgb(
    sw: np.ndarray,
    swy: np.ndarray,
    k: int,
    *,
    additive: bool,
    num_boost_round: int,
    max_depth: int,
    learning_rate: float,
    reg_lambda: float,
    min_child_weight: float,
    seed: int,
) -> np.ndarray:
    seen = sw > 0
    Xtr = LOCAL_STATES[k][seen]
    ytr = swy[seen] / sw[seen]
    wtr = sw[seen]
    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'max_depth': int(min(max_depth, k)),
        'eta': float(learning_rate),
        'lambda': float(reg_lambda),
        'min_child_weight': float(min_child_weight),
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'nthread': 1,
        'seed': int(seed),
        'verbosity': 0,
    }
    if additive:
        # Each group is a singleton: different predictors cannot interact in one tree path.
        params['interaction_constraints'] = json.dumps([[j] for j in range(k)])
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return booster.predict(xgb.DMatrix(LOCAL_STATES[k]))


def _sse_from_state_predictions(pred: np.ndarray, full_sw: np.ndarray, full_swy: np.ndarray, full_swy2: np.ndarray, mask: int) -> float:
    proj = PROJ[mask]
    k = len(BITS[mask])
    nsub = 1 << k
    sw = np.bincount(proj, weights=full_sw, minlength=nsub).astype(float)
    swy = np.bincount(proj, weights=full_swy, minlength=nsub).astype(float)
    return float(full_swy2.sum() - 2.0 * pred @ swy + (pred * pred) @ sw)


def _z_within_order(values: np.ndarray, orders: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.full_like(values, np.nan, dtype=float)
    mu = np.full_like(values, np.nan, dtype=float)
    sd = np.full_like(values, np.nan, dtype=float)
    for k in range(1, N + 1):
        idx = np.flatnonzero(orders == k)
        v = values[idx]
        muk = float(np.mean(v))
        sdk = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        mu[idx] = muk
        sd[idx] = sdk
        if np.isfinite(sdk) and sdk > 0:
            z[idx] = (v - muk) / sdk
    return z, mu, sd


def _safe_corr(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan, np.nan, np.nan, np.nan, int(m.sum())
    pr, pp = pearsonr(x[m], y[m])
    sr, sp = spearmanr(x[m], y[m])
    return float(pr), float(pp), float(sr), float(sp), int(m.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description='Cross-fitted XGBoost atlas for all BRFSS binary subsets.')
    parser.add_argument('--workdir', type=Path, default=Path(os.environ.get('DTHOI_POC_WORKDIR', Path(__file__).resolve().parent / 'work')))
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--reg-lambda', type=float, default=5.0)
    parser.add_argument('--min-child-weight', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    outdir = workdir / 'framework'
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(workdir / 'selected.pkl')
    atlas_path = outdir / 'atlas_calibrated.csv'
    if not atlas_path.exists():
        raise FileNotFoundError('Run atlas.py/calibrate.py first; atlas_calibrated.csv is required for O-information.')
    atlas = pd.read_csv(atlas_path).set_index('mask')

    Xraw = df[XCOLS].to_numpy(float)
    X = np.full_like(Xraw, np.nan)
    X[Xraw == 1] = 1.0
    X[Xraw == 2] = 0.0
    Y = df[YCOL].to_numpy(float) / 100.0
    W = df[WCOL].to_numpy(float)
    Y[(~np.isfinite(Y)) | (Y < 12) | (Y >= 100)] = np.nan
    W[(~np.isfinite(W)) | (W <= 0)] = np.nan
    xok = np.all(np.isfinite(X), axis=1) & np.isfinite(W)
    x_idx = np.flatnonzero(xok)
    pred_idx = x_idx[np.isfinite(Y[x_idx])]

    outer_all = make_fold_ids(len(x_idx), 5, SEED + 100)
    outer_by_row = np.full(len(df), -1, dtype=np.int16)
    outer_by_row[x_idx] = outer_all
    fold_pred = outer_by_row[pred_idx]
    codes = encode_rows(X[pred_idx])
    y = Y[pred_idx]
    w = W[pred_idx]
    tss = weighted_r2_tss(y, w)

    checkpoint = outdir / 'xgb_checkpoint.npz'
    if checkpoint.exists():
        cp = np.load(checkpoint)
        sse_full = cp['sse_full']
        sse_add = cp['sse_add']
        start_fold = int(cp['next_fold'])
        print(f'Resuming XGB atlas at outer fold {start_fold + 1}/5', flush=True)
    else:
        sse_full = np.zeros(NST, dtype=float)
        sse_add = np.zeros(NST, dtype=float)
        sse_full[0] = sse_add[0] = np.nan
        start_fold = 0

    for f in range(start_fold, 5):
        tr = fold_pred != f
        te = fold_pred == f
        tr_stats = _full_state_stats(codes[tr], y[tr], w[tr])
        te_stats = _full_state_stats(codes[te], y[te], w[te])
        common = dict(
            num_boost_round=args.rounds,
            max_depth=args.max_depth,
            learning_rate=args.eta,
            reg_lambda=args.reg_lambda,
            min_child_weight=args.min_child_weight,
            seed=SEED + 5000 + f,
        )
        def fit_mask(mask: int):
            k = len(BITS[mask])
            sw, swy = _aggregate_subset(tr_stats[0], tr_stats[1], mask)
            p_add = _fit_predict_xgb(sw, swy, k, additive=True, **common)
            if k == 1:
                p_full = p_add.copy()
            else:
                p_full = _fit_predict_xgb(sw, swy, k, additive=False, **common)
            return (
                mask,
                _sse_from_state_predictions(p_add, *te_stats, mask),
                _sse_from_state_predictions(p_full, *te_stats, mask),
            )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for mask, sa, sf in pool.map(fit_mask, range(1, NST), chunksize=8):
                sse_add[mask] += sa
                sse_full[mask] += sf
        np.savez(checkpoint, sse_full=sse_full, sse_add=sse_add, next_fold=f + 1)
        print(f'XGB outer fold {f+1}/5 complete', flush=True)

    r2_add = 1.0 - sse_add / tss
    r2_full = 1.0 - sse_full / tss
    delta = r2_full - r2_add
    masks = np.arange(1, NST, dtype=int)
    orders = np.array([len(BITS[m]) for m in masks], dtype=int)
    zp, mu_p, sd_p = _z_within_order(r2_full[masks], orders)
    zd, mu_d, sd_d = _z_within_order(delta[masks], orders)

    rows = []
    for i, mask in enumerate(masks):
        rows.append({
            'mask': mask,
            'order': orders[i],
            'variables': ' | '.join(XCOLS[j] for j in BITS[mask]),
            'Omega_bits_fullX': float(atlas.loc[mask, 'Omega_bits_fullX']) if mask in atlas.index else np.nan,
            'R2_xgb_additive_crossfit': r2_add[mask],
            'R2_xgb_full_crossfit': r2_full[mask],
            'delta_int_xgb': delta[mask],
            'mu_P_size_null': mu_p[i],
            'sd_P_size_null': sd_p[i],
            'Z_P': zp[i],
            'mu_delta_size_null': mu_d[i],
            'sd_delta_size_null': sd_d[i],
            'Z_delta': zd[i],
        })
    out = pd.DataFrame(rows)
    z_omega, _, _ = _z_within_order(out['Omega_bits_fullX'].to_numpy(float), out['order'].to_numpy(int))
    out['Z_Omega_within_order'] = z_omega
    out.to_csv(outdir / 'xgb_atlas.csv', index=False)

    by_order = out.groupby('order').agg(
        n=('mask', 'size'),
        mean_P=('R2_xgb_full_crossfit', 'mean'),
        sd_P=('R2_xgb_full_crossfit', 'std'),
        mean_delta=('delta_int_xgb', 'mean'),
        sd_delta=('delta_int_xgb', 'std'),
        mean_Omega=('Omega_bits_fullX', 'mean'),
    ).reset_index()
    by_order.to_csv(outdir / 'xgb_by_order.csv', index=False)

    qframe = out[(out['order'] >= 3) & (out['order'] <= 10) & out['Omega_bits_fullX'].notna()].copy()
    qframe['omega_quintile_within_order'] = qframe.groupby('order')['Omega_bits_fullX'].transform(
        lambda values: pd.qcut(values.rank(method='first'), 5, labels=False) + 1
    )
    quintiles = qframe.groupby('omega_quintile_within_order').agg(
        n=('mask', 'size'),
        mean_Z_P=('Z_P', 'mean'),
        mean_Z_delta=('Z_delta', 'mean'),
        median_Z_delta=('Z_delta', 'median'),
        mean_delta_int=('delta_int_xgb', 'mean'),
    ).reset_index()
    quintiles.to_csv(outdir / 'xgb_omega_quintiles.csv', index=False)

    # Integrate directly into the current calibrated atlas.
    base = pd.read_csv(atlas_path)
    xgb_cols = ['R2_xgb_additive_crossfit','R2_xgb_full_crossfit','delta_int_xgb','mu_P_size_null','sd_P_size_null','Z_P','mu_delta_size_null','sd_delta_size_null','Z_delta','Z_Omega_within_order']
    base = base.drop(columns=[c for c in xgb_cols if c in base.columns])
    merged = base.merge(
        out[['mask','R2_xgb_additive_crossfit','R2_xgb_full_crossfit','delta_int_xgb','mu_P_size_null','sd_P_size_null','Z_P','mu_delta_size_null','sd_delta_size_null','Z_delta','Z_Omega_within_order']],
        on='mask', how='left', validate='one_to_one'
    )
    merged.to_csv(outdir / 'atlas_calibrated.csv', index=False)

    rel_rows = []
    for metric in ['R2_xgb_full_crossfit','delta_int_xgb','Z_P','Z_delta']:
        pr, pp, sr, sp, n = _safe_corr(out['Omega_bits_fullX'].to_numpy(), out[metric].to_numpy())
        rel_rows.append({'scope':'pooled','order':np.nan,'metric':metric,'n':n,'pearson_r':pr,'pearson_p':pp,'spearman_rho':sr,'spearman_p':sp})
        for k in range(3, N):
            sub = out[out.order == k]
            pr, pp, sr, sp, n = _safe_corr(sub['Omega_bits_fullX'].to_numpy(), sub[metric].to_numpy())
            rel_rows.append({'scope':'within_order','order':k,'metric':metric,'n':n,'pearson_r':pr,'pearson_p':pp,'spearman_rho':sr,'spearman_p':sp})
    for metric in ['Z_P', 'Z_delta']:
        pr, pp, sr, sp, n = _safe_corr(out['Z_Omega_within_order'].to_numpy(), out[metric].to_numpy())
        rel_rows.append({
            'scope': 'size_adjusted', 'order': np.nan, 'metric': metric, 'n': n,
            'pearson_r': pr, 'pearson_p': pp, 'spearman_rho': sr, 'spearman_p': sp,
        })
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(outdir / 'xgb_omega_relationships.csv', index=False)

    summary = {
        'model': {
            'booster':'gbtree','objective':'reg:squarederror','tree_method':'hist',
            'num_boost_round':args.rounds,'max_depth':args.max_depth,'eta':args.eta,
            'reg_lambda':args.reg_lambda,'min_child_weight':args.min_child_weight,
            'additive_definition':'same model with singleton interaction_constraints; only difference from full model',
        },
        'validation': {'folds':5,'fold_seed':SEED+100,'same_folds_as_framework':True},
        'size_null': 'Exact empirical distribution over all subsets of the same size k; uniform random k-subset null, no Monte Carlo sampling.',
        'n_subsets': int(len(out)),
        'interpretation': {
            'P': 'cross-fitted R2 of XGB-full',
            'delta_int': 'cross-fitted R2(XGB-full) - R2(XGB-additive)',
            'Z_P': 'standardized against the exact uniform distribution over all subsets of the same size',
            'Z_delta': 'interaction gain standardized against the exact uniform distribution over all subsets of the same size',
            'size_null_is_not_significance_test': True,
        },
    }
    (outdir / 'xgb_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if checkpoint.exists():
        checkpoint.unlink()
    print(outdir / 'xgb_atlas.csv')

if __name__ == '__main__':
    main()
