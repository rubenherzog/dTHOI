"""Targeted finite-sample O-information benchmark on NLB MC_Maze_Small.

This exploratory script treats O-information itself as the estimand. It tests
(1) the paper's first-order bias correction applied directly to empirical O,
(2) direct extrapolation of O versus 1/N, and (3) uncertainty estimated from
empirical local O with contiguous temporal blocks. No dTHOI core code changes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmark_mc_maze import SEED, _write_csv
from benchmark_mc_maze_stability import (
    N_SETS,
    ORDERS,
    _comparison,
    _dataset_at_20_ms,
    _make_common_sets,
)
from dthoi import information_measures, prepare_data

FIT_SIZES = (2048, 4096, 8192)
HALF_FIT_SIZES = (2048, 4096)
BLOCK_SIZES = (50, 100, 250)  # 1, 2, 5 s at 20 ms/bin.


def _o_values(data: np.ndarray, subsets: torch.Tensor, estimator: str = "empirical") -> np.ndarray:
    """Return one O-information value per variable set in bits."""
    result = information_measures(prepare_data(data), subsets, entropy_estimator=estimator)
    return result.o_information[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)


def _support_count(data: np.ndarray, columns: np.ndarray) -> int:
    """Count observed binary states for one variable subset."""
    selected = data[:, columns].astype(np.int64, copy=False)
    weights = np.left_shift(np.int64(1), np.arange(selected.shape[1], dtype=np.int64))
    codes = selected @ weights
    return int(np.unique(codes).size)


def _direct_observed_mm_correction(data: np.ndarray, subsets: torch.Tensor) -> np.ndarray:
    """Return the direct first-order O bias correction using observed supports."""
    n_samples = data.shape[0]
    rows = subsets.detach().cpu().numpy().astype(np.int64, copy=False)
    corrections = np.empty(len(rows), dtype=np.float64)
    denominator = 2.0 * n_samples * math.log(2.0)
    for row_index, variables in enumerate(rows):
        order = len(variables)
        numerator = (order - 2) * (_support_count(data, variables) - 1)
        for leave_out in range(order):
            singleton = variables[leave_out : leave_out + 1]
            reduced = np.delete(variables, leave_out)
            numerator += _support_count(data, singleton) - _support_count(data, reduced)
        corrections[row_index] = numerator / denominator
    return corrections


def _nominal_binary_correction(order: int, n_samples: int) -> float:
    """Return the paper correction if every binary state is treated as possible."""
    numerator = (order - 4) * (2 ** (order - 1)) + order + 2
    return numerator / (2.0 * n_samples * math.log(2.0))


def _fit_intercept(sample_sizes: list[int], vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fit O(N)=O_inf+a/N independently for aligned variable sets."""
    x = 1.0 / np.asarray(sample_sizes, dtype=np.float64)
    y = np.stack(vectors, axis=0)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = np.sum((y - fitted) ** 2, axis=0)
    centered = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.where(centered > 0, 1.0 - residual / centered, np.nan)
    return coefficients[0], r2


def _method_summary(method: str, order: int, values: np.ndarray, **extra) -> dict:
    """Summarize one aligned O vector."""
    finite = np.isfinite(values)
    row = {
        "method": method,
        "order": order,
        "variable_sets": int(finite.sum()),
        "o_mean_bits": float(np.mean(values[finite])),
        "o_median_bits": float(np.median(values[finite])),
        "o_fraction_positive": float(np.mean(values[finite] > 0)),
    }
    row.update(extra)
    return row


def _run_full_and_extrapolation(binary: np.ndarray, sets_by_order: dict[int, torch.Tensor], output_dir: Path) -> None:
    """Compare plug-in, direct correction, and direct 1/N extrapolation."""
    rng = np.random.default_rng(SEED + 505)
    permutation = rng.permutation(binary.shape[0])
    rows: list[dict] = []
    equivalence_rows: list[dict] = []

    for order, subsets in sets_by_order.items():
        full_plugin = _o_values(binary, subsets)
        correction = _direct_observed_mm_correction(binary, subsets)
        direct_mm = full_plugin + correction
        dthoi_mm = _o_values(binary, subsets, "miller_madow")
        nominal = full_plugin + _nominal_binary_correction(order, binary.shape[0])

        sizes = [n for n in FIT_SIZES if n < binary.shape[0]] + [binary.shape[0]]
        vectors = [_o_values(binary[permutation[:n]], subsets) for n in sizes[:-1]] + [full_plugin]
        extrapolated, r2 = _fit_intercept(sizes, vectors)

        for row in (
            _method_summary("plugin", order, full_plugin),
            _method_summary(
                "direct_mm_observed", order, direct_mm,
                mean_correction_bits=float(np.mean(correction)),
            ),
            _method_summary(
                "direct_mm_nominal_binary", order, nominal,
                mean_correction_bits=_nominal_binary_correction(order, binary.shape[0]),
            ),
            _method_summary(
                "o_extrapolated_1_over_n", order, extrapolated,
                fit_sizes=";".join(map(str, sizes)),
                median_fit_r2=float(np.nanmedian(r2)),
                fraction_fit_r2_ge_0_9=float(np.nanmean(r2 >= 0.9)),
            ),
        ):
            rows.append(row)
            print("O_METHOD", json.dumps(row, sort_keys=True))

        eq = {
            "order": order,
            "max_abs_direct_mm_vs_dthoi_mm_bits": float(np.max(np.abs(direct_mm - dthoi_mm))),
            "mean_abs_direct_mm_vs_dthoi_mm_bits": float(np.mean(np.abs(direct_mm - dthoi_mm))),
            "extrapolated_vs_plugin_spearman": _comparison(extrapolated, full_plugin)["spearman"],
            "extrapolated_vs_plugin_sign_agreement": _comparison(extrapolated, full_plugin)["sign_agreement"],
            "extrapolated_minus_plugin_mean_bits": float(np.mean(extrapolated - full_plugin)),
        }
        equivalence_rows.append(eq)
        print("O_EQUIVALENCE", json.dumps(eq, sort_keys=True))

    _write_csv(output_dir / "o_direct_bias_methods.csv", rows)
    _write_csv(output_dir / "o_direct_bias_equivalence.csv", equivalence_rows)


def _run_split_half(binary: np.ndarray, sets_by_order: dict[int, torch.Tensor], output_dir: Path) -> None:
    """Compare split-half reproducibility of direct O estimators."""
    rng = np.random.default_rng(SEED + 606)
    permutation = rng.permutation(binary.shape[0])
    midpoint = len(permutation) // 2
    halves = [binary[permutation[:midpoint]], binary[permutation[midpoint : 2 * midpoint]]]
    rows: list[dict] = []

    for order, subsets in sets_by_order.items():
        estimates: dict[str, list[np.ndarray]] = {"plugin": [], "direct_mm_observed": [], "o_extrapolated_1_over_n": []}
        fit_r2: list[np.ndarray] = []
        for half in halves:
            plugin = _o_values(half, subsets)
            direct_mm = plugin + _direct_observed_mm_correction(half, subsets)
            half_rng = np.random.default_rng(SEED + 700 + order + len(estimates["plugin"]))
            half_perm = half_rng.permutation(len(half))
            sizes = [n for n in HALF_FIT_SIZES if n < len(half)] + [len(half)]
            vectors = [_o_values(half[half_perm[:n]], subsets) for n in sizes[:-1]] + [plugin]
            extrapolated, r2 = _fit_intercept(sizes, vectors)
            estimates["plugin"].append(plugin)
            estimates["direct_mm_observed"].append(direct_mm)
            estimates["o_extrapolated_1_over_n"].append(extrapolated)
            fit_r2.append(r2)

        for method, (left, right) in estimates.items():
            cmp = _comparison(left, right)
            row = {
                "method": method,
                "order": order,
                "samples_per_half": midpoint,
                "o_spearman_between_halves": cmp["spearman"],
                "o_pearson_between_halves": cmp["pearson"],
                "o_sign_agreement_between_halves": cmp["sign_agreement"],
                "o_mae_between_halves_bits": cmp["mae"],
                "o_mean_half_a_bits": float(np.mean(left)),
                "o_mean_half_b_bits": float(np.mean(right)),
            }
            if method == "o_extrapolated_1_over_n":
                row["median_fit_r2"] = float(np.nanmedian(np.concatenate(fit_r2)))
            rows.append(row)
            print("O_SPLIT_HALF", json.dumps(row, sort_keys=True))

    _write_csv(output_dir / "o_direct_bias_split_half.csv", rows)


def _block_se(local_o: np.ndarray, block_size: int) -> np.ndarray:
    """Estimate SE(mean O) from non-overlapping contiguous block means."""
    n_sets, n_samples = local_o.shape
    n_blocks = n_samples // block_size
    trimmed = local_o[:, : n_blocks * block_size]
    block_means = trimmed.reshape(n_sets, n_blocks, block_size).mean(axis=2)
    return block_means.std(axis=1, ddof=1) / math.sqrt(n_blocks)


def _run_local_uncertainty(binary: np.ndarray, sets_by_order: dict[int, torch.Tensor], output_dir: Path) -> None:
    """Estimate empirical O uncertainty from sample-resolved O using time blocks."""
    prepared = prepare_data(binary)
    rows: list[dict] = []
    for order, subsets in sets_by_order.items():
        result = information_measures(prepared, subsets, entropy_estimator="empirical", local_values=True)
        global_o = result.o_information[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        local_o = result.local.o_information[0].detach().cpu().numpy().astype(np.float64, copy=False)
        recovery = np.max(np.abs(local_o.mean(axis=1) - global_o))
        for block_size in BLOCK_SIZES:
            se = _block_se(local_o, block_size)
            z = np.abs(global_o) / se
            ci_excludes_zero = np.abs(global_o) > 1.96 * se
            row = {
                "order": order,
                "block_bins": block_size,
                "block_seconds": 0.020 * block_size,
                "blocks": binary.shape[0] // block_size,
                "median_se_bits": float(np.median(se)),
                "median_abs_o_over_se": float(np.median(z)),
                "fraction_95pct_ci_excludes_zero": float(np.mean(ci_excludes_zero)),
                "fraction_significant_negative": float(np.mean((global_o < -1.96 * se))),
                "fraction_significant_positive": float(np.mean((global_o > 1.96 * se))),
                "max_local_mean_recovery_error_bits": float(recovery),
            }
            rows.append(row)
            print("O_UNCERTAINTY", json.dumps(row, sort_keys=True))
    _write_csv(output_dir / "o_local_block_uncertainty.csv", rows)


def run(nwb_path: Path, output_dir: Path) -> None:
    """Run the targeted direct-O finite-sample analyses."""
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, binary, active_pool = _dataset_at_20_ms(nwb_path)
    sets_by_order = _make_common_sets(active_pool)
    _run_full_and_extrapolation(binary, sets_by_order, output_dir)
    _run_split_half(binary, sets_by_order, output_dir)
    _run_local_uncertainty(binary, sets_by_order, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("benchmark-results"), type=Path)
    args = parser.parse_args()
    run(args.nwb_path, args.output_dir)
