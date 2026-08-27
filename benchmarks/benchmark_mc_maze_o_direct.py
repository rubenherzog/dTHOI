"""Targeted direct O-information convergence and uncertainty benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_mc_maze import SEED, _write_csv
from benchmark_mc_maze_stability import _comparison, _dataset_at_20_ms, _make_common_sets
from dthoi import information_measures, prepare_data

HALF_FIT_SIZES = (2048, 4096)
FULL_FIT_SIZES = (2048, 4096, 8192)
BLOCK_BINS = (50, 100, 250)  # 1, 2, 5 s at 20 ms/bin.


def _o_values(binary: np.ndarray, subsets: torch.Tensor) -> np.ndarray:
    """Return empirical O-information for aligned variable sets."""
    result = information_measures(
        prepare_data(binary), subsets, entropy_estimator="empirical"
    )
    return result.o_information[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)


def _linear_extrapolation(
    sample_sizes: list[int], curves: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit O(N)=O_inf+a/N independently for every variable set."""
    x = 1.0 / np.asarray(sample_sizes, dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x))
    coefficients, *_ = np.linalg.lstsq(design, curves, rcond=None)
    fitted = design @ coefficients
    residual = curves - fitted
    ss_res = np.sum(residual**2, axis=0)
    centered = curves - curves.mean(axis=0, keepdims=True)
    ss_tot = np.sum(centered**2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return coefficients[0], r2


def _curves(
    binary: np.ndarray,
    subsets: torch.Tensor,
    requested_sizes: tuple[int, ...],
) -> tuple[list[int], np.ndarray]:
    """Calculate nested empirical O curves ending at the full supplied sample."""
    sizes = [n for n in requested_sizes if n < len(binary)] + [len(binary)]
    values = np.stack([_o_values(binary[:n], subsets) for n in sizes], axis=0)
    return sizes, values


def _run_extrapolation(
    binary: np.ndarray,
    sets_by_order: dict[int, torch.Tensor],
    output_dir: Path,
) -> None:
    """Test whether direct 1/N extrapolation improves O reproducibility."""
    permutation = np.random.default_rng(SEED + 303).permutation(binary.shape[0])
    midpoint = len(permutation) // 2
    half_a = binary[permutation[:midpoint]]
    half_b = binary[permutation[midpoint : 2 * midpoint]]
    full = binary[np.random.default_rng(SEED + 202).permutation(binary.shape[0])]

    summary_rows: list[dict] = []
    per_set_rows: list[dict] = []
    for order, subsets in sets_by_order.items():
        half_sizes_a, curves_a = _curves(half_a, subsets, HALF_FIT_SIZES)
        half_sizes_b, curves_b = _curves(half_b, subsets, HALF_FIT_SIZES)
        extrapolated_a, r2_a = _linear_extrapolation(half_sizes_a, curves_a)
        extrapolated_b, r2_b = _linear_extrapolation(half_sizes_b, curves_b)
        raw_a = curves_a[-1]
        raw_b = curves_b[-1]

        full_sizes, full_curves = _curves(full, subsets, FULL_FIT_SIZES)
        extrapolated_full, full_r2 = _linear_extrapolation(full_sizes, full_curves)
        raw_full = full_curves[-1]

        raw_cmp = _comparison(raw_a, raw_b)
        extrapolated_cmp = _comparison(extrapolated_a, extrapolated_b)
        full_cmp = _comparison(extrapolated_full, raw_full)
        summary = {
            "order": order,
            "variable_sets": int(subsets.shape[0]),
            "samples_per_half": midpoint,
            "half_fit_sizes": ";".join(map(str, half_sizes_a)),
            "full_fit_sizes": ";".join(map(str, full_sizes)),
            "raw_half_mae_bits": raw_cmp["mae"],
            "raw_half_pearson": raw_cmp["pearson"],
            "raw_half_spearman": raw_cmp["spearman"],
            "raw_half_sign_agreement": raw_cmp["sign_agreement"],
            "extrapolated_half_mae_bits": extrapolated_cmp["mae"],
            "extrapolated_half_pearson": extrapolated_cmp["pearson"],
            "extrapolated_half_spearman": extrapolated_cmp["spearman"],
            "extrapolated_half_sign_agreement": extrapolated_cmp["sign_agreement"],
            "median_half_fit_r2": float(np.nanmedian(np.concatenate((r2_a, r2_b)))),
            "full_empirical_mean_bits": float(np.mean(raw_full)),
            "full_extrapolated_mean_bits": float(np.mean(extrapolated_full)),
            "full_extrapolation_shift_mean_bits": float(np.mean(extrapolated_full - raw_full)),
            "full_extrapolation_sign_change_fraction": float(
                np.mean(np.signbit(extrapolated_full) != np.signbit(raw_full))
            ),
            "full_extrapolated_vs_empirical_spearman": full_cmp["spearman"],
            "median_full_fit_r2": float(np.nanmedian(full_r2)),
            "fraction_full_fit_r2_ge_0_9": float(np.nanmean(full_r2 >= 0.9)),
        }
        summary_rows.append(summary)
        print("DIRECT_O_EXTRAPOLATION", json.dumps(summary, sort_keys=True))

        for index in range(len(raw_full)):
            per_set_rows.append(
                {
                    "order": order,
                    "set_index": index,
                    "raw_half_a_bits": float(raw_a[index]),
                    "raw_half_b_bits": float(raw_b[index]),
                    "extrapolated_half_a_bits": float(extrapolated_a[index]),
                    "extrapolated_half_b_bits": float(extrapolated_b[index]),
                    "fit_r2_half_a": float(r2_a[index]),
                    "fit_r2_half_b": float(r2_b[index]),
                    "full_empirical_bits": float(raw_full[index]),
                    "full_extrapolated_bits": float(extrapolated_full[index]),
                    "full_fit_r2": float(full_r2[index]),
                }
            )

    _write_csv(output_dir / "direct_o_extrapolation_summary.csv", summary_rows)
    _write_csv(output_dir / "direct_o_extrapolation_per_set.csv", per_set_rows)


def _run_local_block_uncertainty(
    binary: np.ndarray,
    sets_by_order: dict[int, torch.Tensor],
    output_dir: Path,
) -> None:
    """Estimate O uncertainty from local O using contiguous temporal blocks."""
    prepared = prepare_data(binary)
    summary_rows: list[dict] = []
    per_set_rows: list[dict] = []
    for order, subsets in sets_by_order.items():
        result = information_measures(
            prepared,
            subsets,
            entropy_estimator="empirical",
            local_values=True,
        )
        local_o = (
            result.local.o_information[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        global_o = (
            result.o_information[:, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        iid_se = local_o.std(axis=1, ddof=1) / np.sqrt(local_o.shape[1])

        for block_bins in BLOCK_BINS:
            n_blocks = local_o.shape[1] // block_bins
            trimmed = local_o[:, : n_blocks * block_bins]
            block_means = trimmed.reshape(len(subsets), n_blocks, block_bins).mean(axis=2)
            block_se = block_means.std(axis=1, ddof=1) / np.sqrt(n_blocks)
            inflation = block_se / iid_se
            excludes_zero = np.abs(global_o) > 1.96 * block_se
            summary = {
                "order": order,
                "variable_sets": int(subsets.shape[0]),
                "block_bins": block_bins,
                "block_seconds": 0.020 * block_bins,
                "blocks": n_blocks,
                "median_iid_se_bits": float(np.median(iid_se)),
                "median_block_se_bits": float(np.median(block_se)),
                "median_se_inflation_vs_iid": float(np.median(inflation)),
                "fraction_95pct_ci_excludes_zero": float(np.mean(excludes_zero)),
                "fraction_significant_negative": float(
                    np.mean(global_o < -1.96 * block_se)
                ),
                "fraction_significant_positive": float(
                    np.mean(global_o > 1.96 * block_se)
                ),
            }
            summary_rows.append(summary)
            print("LOCAL_O_BLOCK_SE", json.dumps(summary, sort_keys=True))

            for index in range(len(subsets)):
                per_set_rows.append(
                    {
                        "order": order,
                        "set_index": index,
                        "o_bits": float(global_o[index]),
                        "block_bins": block_bins,
                        "block_seconds": 0.020 * block_bins,
                        "iid_se_bits": float(iid_se[index]),
                        "block_se_bits": float(block_se[index]),
                        "se_inflation_vs_iid": float(inflation[index]),
                        "ci95_excludes_zero": bool(excludes_zero[index]),
                    }
                )

    _write_csv(output_dir / "local_o_block_uncertainty_summary.csv", summary_rows)
    _write_csv(output_dir / "local_o_block_uncertainty_per_set.csv", per_set_rows)


def run(nwb_path: Path, output_dir: Path) -> None:
    """Run only the direct-O analyses that add empirical evidence."""
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
