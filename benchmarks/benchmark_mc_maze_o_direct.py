"""Targeted direct O-information bias and uncertainty benchmark on MC_Maze_Small."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_mc_maze import SEED, _write_csv
from benchmark_mc_maze_stability import _comparison, _dataset_at_20_ms, _make_common_sets
from dthoi import information_measures, prepare_data

FIT_SIZES = (1024, 2048, 4096, 7341)
FULL_FIT_SIZES = (2048, 4096, 8192, 14683)
BLOCK_LENGTHS = (1, 5, 10, 25, 50)


def _o_values(binary: np.ndarray, subsets: torch.Tensor) -> np.ndarray:
    result = information_measures(prepare_data(binary), subsets, entropy_estimator="empirical")
    return result.o_information[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)


def _linear_extrapolation(sample_sizes: list[int], curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit O(N)=O_inf+a/N independently for every variable set."""
    x = 1.0 / np.asarray(sample_sizes, dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x))
    coef, *_ = np.linalg.lstsq(design, curves, rcond=None)
    fitted = design @ coef
    residual = curves - fitted
    ss_res = np.sum(residual**2, axis=0)
    centered = curves - curves.mean(axis=0, keepdims=True)
    ss_tot = np.sum(centered**2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return coef[0], r2


def _half_curves(binary_half: np.ndarray, subsets: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = [n for n in FIT_SIZES if n <= len(binary_half)]
    curves = np.stack([_o_values(binary_half[:n], subsets) for n in sizes], axis=0)
    intercept, r2 = _linear_extrapolation(sizes, curves)
    return curves[-1], intercept, r2


def _run_extrapolation(binary: np.ndarray, sets_by_order: dict[int, torch.Tensor], output_dir: Path) -> None:
    rng = np.random.default_rng(SEED + 303)
    permutation = rng.permutation(binary.shape[0])
    midpoint = len(permutation) // 2
    half_a = binary[permutation[:midpoint]]
    half_b = binary[permutation[midpoint : 2 * midpoint]]

    summary_rows: list[dict] = []
    per_set_rows: list[dict] = []
    for order, subsets in sets_by_order.items():
        raw_a, inf_a, r2_a = _half_curves(half_a, subsets)
        raw_b, inf_b, r2_b = _half_curves(half_b, subsets)
        raw_cmp = _comparison(raw_a, raw_b)
        inf_cmp = _comparison(inf_a, inf_b)

        full_perm = np.random.default_rng(SEED + 202).permutation(binary.shape[0])
        full_sizes = [n for n in FULL_FIT_SIZES if n <= len(binary)]
        full_curves = np.stack([_o_values(binary[full_perm[:n]], subsets) for n in full_sizes], axis=0)
        full_inf, full_r2 = _linear_extrapolation(full_sizes, full_curves)
        full_raw = _o_values(binary, subsets)
        full_mm = information_measures(
            prepare_data(binary), subsets, entropy_estimator="miller_madow"
        ).o_information[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)

        summary = {
            "order": order,
            "variable_sets": int(subsets.shape[0]),
            "samples_per_half": midpoint,
            "raw_half_mae_bits": raw_cmp["mae"],
            "raw_half_pearson": raw_cmp["pearson"],
            "raw_half_spearman": raw_cmp["spearman"],
            "raw_half_sign_agreement": raw_cmp["sign_agreement"],
            "extrapolated_half_mae_bits": inf_cmp["mae"],
            "extrapolated_half_pearson": inf_cmp["pearson"],
            "extrapolated_half_spearman": inf_cmp["spearman"],
            "extrapolated_half_sign_agreement": inf_cmp["sign_agreement"],
            "median_half_fit_r2": float(np.nanmedian(np.concatenate((r2_a, r2_b)))),
            "full_empirical_mean_bits": float(np.mean(full_raw)),
            "full_miller_madow_mean_bits": float(np.mean(full_mm)),
            "full_extrapolated_mean_bits": float(np.mean(full_inf)),
            "full_extrapolation_shift_mean_bits": float(np.mean(full_inf - full_raw)),
            "full_extrapolation_sign_change_fraction": float(np.mean(np.signbit(full_inf) != np.signbit(full_raw))),
            "full_extrapolated_vs_empirical_spearman": _comparison(full_inf, full_raw)["spearman"],
            "full_extrapolated_vs_miller_spearman": _comparison(full_inf, full_mm)["spearman"],
            "median_full_fit_r2": float(np.nanmedian(full_r2)),
        }
        summary_rows.append(summary)
        print("DIRECT_O_EXTRAPOLATION", json.dumps(summary, sort_keys=True))

        for j in range(len(full_raw)):
            per_set_rows.append({
                "order": order,
                "set_index": j,
                "raw_half_a_bits": float(raw_a[j]),
                "raw_half_b_bits": float(raw_b[j]),
                "extrapolated_half_a_bits": float(inf_a[j]),
                "extrapolated_half_b_bits": float(inf_b[j]),
                "fit_r2_half_a": float(r2_a[j]),
                "fit_r2_half_b": float(r2_b[j]),
                "full_empirical_bits": float(full_raw[j]),
                "full_miller_madow_bits": float(full_mm[j]),
                "full_extrapolated_bits": float(full_inf[j]),
                "full_fit_r2": float(full_r2[j]),
            })

    _write_csv(output_dir / "direct_o_extrapolation_summary.csv", summary_rows)
    _write_csv(output_dir / "direct_o_extrapolation_per_set.csv", per_set_rows)


def _run_local_block_uncertainty(binary: np.ndarray, sets_by_order: dict[int, torch.Tensor], output_dir: Path) -> None:
    prepared = prepare_data(binary)
    rows: list[dict] = []
    per_set_rows: list[dict] = []
    for order, subsets in sets_by_order.items():
        result = information_measures(
            prepared,
            subsets,
            entropy_estimator="empirical",
            local_values=True,
        )
        local_o = result.local.o_information[0].detach().cpu().numpy().astype(np.float64, copy=False)
        global_o = result.o_information[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        iid_se = local_o.std(axis=1, ddof=1) / np.sqrt(local_o.shape[1])

        for block_bins in BLOCK_LENGTHS:
            n_blocks = local_o.shape[1] // block_bins
            trimmed = local_o[:, : n_blocks * block_bins]
            block_means = trimmed.reshape(len(subsets), n_blocks, block_bins).mean(axis=2)
            block_se = block_means.std(axis=1, ddof=1) / np.sqrt(n_blocks)
            inflation = block_se / iid_se
            excludes_zero = np.abs(global_o) > 1.96 * block_se
            row = {
                "order": order,
                "variable_sets": int(subsets.shape[0]),
                "block_bins": block_bins,
                "block_ms": 20 * block_bins,
                "blocks": n_blocks,
                "median_iid_se_bits": float(np.median(iid_se)),
                "median_block_se_bits": float(np.median(block_se)),
                "median_se_inflation_vs_iid": float(np.median(inflation)),
                "fraction_95ci_excludes_zero": float(np.mean(excludes_zero)),
            }
            rows.append(row)
            print("LOCAL_O_BLOCK_SE", json.dumps(row, sort_keys=True))

            for j in range(len(subsets)):
                per_set_rows.append({
                    "order": order,
                    "set_index": j,
                    "o_bits": float(global_o[j]),
                    "block_bins": block_bins,
                    "block_ms": 20 * block_bins,
                    "iid_se_bits": float(iid_se[j]),
                    "block_se_bits": float(block_se[j]),
                    "se_inflation_vs_iid": float(inflation[j]),
                    "ci95_excludes_zero": bool(excludes_zero[j]),
                })

    _write_csv(output_dir / "local_o_block_uncertainty_summary.csv", rows)
    _write_csv(output_dir / "local_o_block_uncertainty_per_set.csv", per_set_rows)


def run(nwb_path: Path, output_dir: Path) -> None:
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, binary, active_pool = _dataset_at_20_ms(nwb_path)
    sets_by_order = _make_common_sets(active_pool)
    _run_extrapolation(binary, sets_by_order, output_dir)
    _run_local_block_uncertainty(binary, sets_by_order, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("benchmark-results"), type=Path)
    args = parser.parse_args()
    run(args.nwb_path, args.output_dir)
