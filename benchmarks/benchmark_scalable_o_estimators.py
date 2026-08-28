"""Compare scalable coherent O-information estimators with dTHOI estimators.

The benchmark has two parts: controlled binary systems with known O-information
and the pinned NLB MC_Maze_Small dataset used by the existing validation branch.
It is deliberately estimator-focused and does not modify the dTHOI core API.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from benchmark_mc_maze import SEED, _write_csv
from benchmark_mc_maze_stability import (
    ANALYSIS_ESTIMATORS,
    _comparison,
    _dataset_at_20_ms,
    _make_common_sets,
)
from coherent_o_estimators import all_coherent_estimators
from dthoi import information_measures, prepare_data

EXISTING_ESTIMATORS = ANALYSIS_ESTIMATORS + ("ansb",)
NEW_ESTIMATORS = (
    "coherent_dirichlet_a1",
    "coherent_dirichlet_eb",
    "coherent_nsb",
)
SYNTHETIC_ORDERS = (8, 16, 32)
SYNTHETIC_SAMPLES = 4096
SYNTHETIC_REPLICATES = 10
SYNTHETIC_SYSTEM_VARIABLES = 128


def _synthetic_data(
    kind: str,
    *,
    replicates: int,
    samples: int,
    order: int,
    system_variables: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Generate binary systems with analytically known O-information."""
    if system_variables < order:
        raise ValueError("system_variables must be at least order.")
    data = rng.integers(
        0,
        2,
        size=(replicates, samples, system_variables),
        dtype=np.uint8,
    )
    if kind == "independent":
        truth = 0.0
    elif kind == "redundant":
        latent = rng.integers(0, 2, size=(replicates, samples, 1), dtype=np.uint8)
        data[:, :, :order] = np.repeat(latent, order, axis=2)
        truth = float(order - 2)
    elif kind == "parity":
        first = rng.integers(
            0, 2, size=(replicates, samples, order - 1), dtype=np.uint8
        )
        data[:, :, : order - 1] = first
        data[:, :, order - 1] = np.bitwise_xor.reduce(first, axis=2)
        truth = -float(order - 2)
    else:
        raise ValueError(f"Unknown synthetic system: {kind}")
    return data, truth


def _metric_row(
    *,
    system: str,
    order: int,
    estimator: str,
    truth: float,
    estimates: np.ndarray,
    tc: np.ndarray,
    dtc: np.ndarray,
    seconds: float,
    concentrations: np.ndarray | None = None,
) -> dict:
    """Summarize finite-sample bias and variance against known ground truth."""
    estimates = np.asarray(estimates, dtype=np.float64)
    finite = np.isfinite(estimates)
    kept = estimates[finite]
    errors = kept - truth
    concentration = (
        float(np.median(concentrations[np.isfinite(concentrations)]))
        if concentrations is not None and np.isfinite(concentrations).any()
        else float("nan")
    )
    return {
        "system": system,
        "system_variables": SYNTHETIC_SYSTEM_VARIABLES,
        "order": order,
        "samples": SYNTHETIC_SAMPLES,
        "replicates": SYNTHETIC_REPLICATES,
        "estimator": estimator,
        "truth_o_bits": truth,
        "mean_o_bits": float(np.mean(kept)) if kept.size else float("nan"),
        "bias_bits": float(np.mean(errors)) if kept.size else float("nan"),
        "sd_bits": float(np.std(kept, ddof=0)) if kept.size else float("nan"),
        "rmse_bits": float(np.sqrt(np.mean(errors**2))) if kept.size else float("nan"),
        "sign_accuracy": (
            float(np.mean(np.signbit(kept) == np.signbit(truth)))
            if kept.size and truth != 0
            else float("nan")
        ),
        "finite_fraction": float(np.mean(finite)),
        "fraction_tc_negative": float(np.mean(np.asarray(tc) < -1e-12)),
        "fraction_dtc_negative": float(np.mean(np.asarray(dtc) < -1e-12)),
        "median_concentration": concentration,
        "seconds": seconds,
    }


def _run_synthetic(output_dir: Path) -> None:
    """Compare all estimators on known independent, redundant, and parity laws."""
    rng = np.random.default_rng(SEED + 900)
    subset_by_order = {
        order: torch.arange(order, dtype=torch.long).unsqueeze(0)
        for order in SYNTHETIC_ORDERS
    }
    rows: list[dict] = []

    for order in SYNTHETIC_ORDERS:
        subset = subset_by_order[order]
        for system in ("independent", "redundant", "parity"):
            data, truth = _synthetic_data(
                system,
                replicates=SYNTHETIC_REPLICATES,
                samples=SYNTHETIC_SAMPLES,
                order=order,
                system_variables=SYNTHETIC_SYSTEM_VARIABLES,
                rng=rng,
            )
            selected = data[:, :, :order]
            prepared = prepare_data(selected)

            for estimator in EXISTING_ESTIMATORS:
                start = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = information_measures(
                            prepared,
                            subset,
                            entropy_estimator=estimator,
                        )
                    elapsed = time.perf_counter() - start
                    o = result.o_information[0].detach().cpu().numpy()
                    tc = result.total_correlation[0].detach().cpu().numpy()
                    dtc = result.dual_total_correlation[0].detach().cpu().numpy()
                except Exception:
                    elapsed = time.perf_counter() - start
                    o = np.full(SYNTHETIC_REPLICATES, np.nan)
                    tc = np.full(SYNTHETIC_REPLICATES, np.nan)
                    dtc = np.full(SYNTHETIC_REPLICATES, np.nan)
                row = _metric_row(
                    system=system,
                    order=order,
                    estimator=estimator,
                    truth=truth,
                    estimates=o,
                    tc=tc,
                    dtc=dtc,
                    seconds=elapsed,
                )
                rows.append(row)
                print("SYNTHETIC", json.dumps(row, sort_keys=True))

            new_o = {name: [] for name in NEW_ESTIMATORS}
            new_tc = {name: [] for name in NEW_ESTIMATORS}
            new_dtc = {name: [] for name in NEW_ESTIMATORS}
            new_a = {name: [] for name in NEW_ESTIMATORS}
            start = time.perf_counter()
            for replicate in selected:
                estimates = all_coherent_estimators(replicate)
                for name, estimate in estimates.items():
                    new_o[name].append(estimate.o_information)
                    new_tc[name].append(estimate.total_correlation)
                    new_dtc[name].append(estimate.dual_total_correlation)
                    new_a[name].append(estimate.concentration)
            elapsed_all = time.perf_counter() - start
            for name in NEW_ESTIMATORS:
                row = _metric_row(
                    system=system,
                    order=order,
                    estimator=name,
                    truth=truth,
                    estimates=np.asarray(new_o[name]),
                    tc=np.asarray(new_tc[name]),
                    dtc=np.asarray(new_dtc[name]),
                    seconds=elapsed_all,
                    concentrations=np.asarray(new_a[name]),
                )
                rows.append(row)
                print("SYNTHETIC", json.dumps(row, sort_keys=True))

    _write_csv(output_dir / "scalable_o_synthetic.csv", rows)


def _existing_vector(data: np.ndarray, subsets: torch.Tensor, estimator: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return aligned O, TC, and DTC vectors for one existing estimator."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = information_measures(
            prepare_data(data), subsets, entropy_estimator=estimator
        )
    elapsed = time.perf_counter() - start
    return (
        result.o_information[:, 0].detach().cpu().numpy().astype(np.float64, copy=False),
        result.total_correlation[:, 0].detach().cpu().numpy().astype(np.float64, copy=False),
        result.dual_total_correlation[:, 0].detach().cpu().numpy().astype(np.float64, copy=False),
        elapsed,
    )


def _new_vectors(data: np.ndarray, subsets: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Evaluate all coherent estimators while sharing counts within each set."""
    arrays = subsets.detach().cpu().numpy()
    o = {name: [] for name in NEW_ESTIMATORS}
    tc = {name: [] for name in NEW_ESTIMATORS}
    dtc = {name: [] for name in NEW_ESTIMATORS}
    concentration = {name: [] for name in NEW_ESTIMATORS}
    start = time.perf_counter()
    for variables in arrays:
        estimates = all_coherent_estimators(data[:, variables])
        for name, estimate in estimates.items():
            o[name].append(estimate.o_information)
            tc[name].append(estimate.total_correlation)
            dtc[name].append(estimate.dual_total_correlation)
            concentration[name].append(estimate.concentration)
    elapsed = time.perf_counter() - start
    convert = lambda mapping: {key: np.asarray(value, dtype=np.float64) for key, value in mapping.items()}
    return convert(o), convert(tc), convert(dtc), convert(concentration), elapsed


def _real_summary_row(
    estimator: str,
    order: int,
    o: np.ndarray,
    tc: np.ndarray,
    dtc: np.ndarray,
    seconds: float,
    concentration: np.ndarray | None = None,
) -> dict:
    """Summarize one full-data MC_Maze estimator vector."""
    finite = np.isfinite(o)
    kept = o[finite]
    return {
        "estimator": estimator,
        "order": order,
        "variable_sets": int(len(o)),
        "finite_fraction": float(np.mean(finite)),
        "o_mean_bits": float(np.mean(kept)) if kept.size else float("nan"),
        "o_std_bits": float(np.std(kept)) if kept.size else float("nan"),
        "o_fraction_positive": float(np.mean(kept > 0)) if kept.size else float("nan"),
        "fraction_tc_negative": float(np.mean(tc < -1e-12)),
        "fraction_dtc_negative": float(np.mean(dtc < -1e-12)),
        "median_concentration": (
            float(np.median(concentration[np.isfinite(concentration)]))
            if concentration is not None and np.isfinite(concentration).any()
            else float("nan")
        ),
        "seconds": seconds,
    }


def _run_mc_maze(nwb_path: Path, output_dir: Path) -> None:
    """Compare full-data values and random split-half reproducibility on MC_Maze."""
    _, binary, active_pool = _dataset_at_20_ms(nwb_path)
    sets_by_order = _make_common_sets(active_pool)
    full_rows: list[dict] = []
    split_rows: list[dict] = []

    rng = np.random.default_rng(SEED + 901)
    permutation = rng.permutation(binary.shape[0])
    midpoint = len(permutation) // 2
    halves = [
        binary[permutation[:midpoint]],
        binary[permutation[midpoint : 2 * midpoint]],
    ]

    for order, subsets in sets_by_order.items():
        full_vectors: dict[str, np.ndarray] = {}
        half_vectors: dict[str, list[np.ndarray]] = {}
        for estimator in EXISTING_ESTIMATORS:
            try:
                o, tc, dtc, elapsed = _existing_vector(binary, subsets, estimator)
                full_vectors[estimator] = o
                full_rows.append(_real_summary_row(estimator, order, o, tc, dtc, elapsed))
                half_vectors[estimator] = [
                    _existing_vector(half, subsets, estimator)[0] for half in halves
                ]
            except Exception:
                continue

        new_o, new_tc, new_dtc, new_a, new_elapsed = _new_vectors(binary, subsets)
        for estimator in NEW_ESTIMATORS:
            full_vectors[estimator] = new_o[estimator]
            full_rows.append(
                _real_summary_row(
                    estimator,
                    order,
                    new_o[estimator],
                    new_tc[estimator],
                    new_dtc[estimator],
                    new_elapsed,
                    new_a[estimator],
                )
            )
        half_new = [_new_vectors(half, subsets)[0] for half in halves]
        for estimator in NEW_ESTIMATORS:
            half_vectors[estimator] = [half_new[0][estimator], half_new[1][estimator]]

        empirical_full = full_vectors.get("empirical")
        for estimator, vectors in half_vectors.items():
            cmp = _comparison(vectors[0], vectors[1])
            row = {
                "estimator": estimator,
                "order": order,
                "samples_per_half": midpoint,
                "spearman_between_halves": cmp["spearman"],
                "pearson_between_halves": cmp["pearson"],
                "sign_agreement_between_halves": cmp["sign_agreement"],
                "mae_between_halves_bits": cmp["mae"],
                "finite_values": cmp["finite_values"],
                "spearman_full_vs_empirical": (
                    _comparison(full_vectors[estimator], empirical_full)["spearman"]
                    if empirical_full is not None and estimator in full_vectors
                    else float("nan")
                ),
            }
            split_rows.append(row)
            print("MC_SPLIT", json.dumps(row, sort_keys=True))

    for row in full_rows:
        print("MC_FULL", json.dumps(row, sort_keys=True))
    _write_csv(output_dir / "scalable_o_mc_maze_full.csv", full_rows)
    _write_csv(output_dir / "scalable_o_mc_maze_split.csv", split_rows)


def _run_scaling(output_dir: Path) -> None:
    """Exercise the new estimators inside a 500-variable ambient system."""
    rng = np.random.default_rng(SEED + 902)
    samples = 2048
    variables = 500
    data = rng.integers(0, 2, size=(samples, variables), dtype=np.uint8)
    rows: list[dict] = []
    for order in (8, 16, 32, 48, 64, 96):
        start = time.perf_counter()
        estimates = all_coherent_estimators(data[:, :order])
        elapsed = time.perf_counter() - start
        for name, estimate in estimates.items():
            row = {
                "system_variables": variables,
                "order": order,
                "samples": samples,
                "estimator": name,
                "seconds_all_three_shared_counts": elapsed,
                "o_bits": estimate.o_information,
                "concentration": estimate.concentration,
            }
            rows.append(row)
            print("SCALING", json.dumps(row, sort_keys=True))
    _write_csv(output_dir / "scalable_o_500_variable_scaling.csv", rows)


def run(nwb_path: Path, output_dir: Path) -> None:
    """Run controlled, real-data, and ambient-dimensionality comparisons."""
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_synthetic(output_dir)
    _run_mc_maze(nwb_path, output_dir)
    _run_scaling(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("benchmark-results"), type=Path)
    args = parser.parse_args()
    run(args.nwb_path, args.output_dir)
