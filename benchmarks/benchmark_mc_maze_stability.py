"""Stability and isolated-memory benchmark for NLB MC_Maze_Small.

This exploratory benchmark complements ``benchmark_mc_maze.py`` without
changing dTHOI's core. It tests whether higher-order estimates on real neural
spiking data are stable to sample size, spike-bin width, and split-half
resampling, and it measures global-analysis memory growth in fresh subprocesses.

The current dTHOI binary contract is respected by representing each neuron/bin
as 1 when at least one spike occurs and 0 otherwise. All information quantities
are reported in bits. ANSB is intentionally excluded from these stability
analyses because the first MC_Maze benchmark showed that its severe-
undersampling assumption is violated for the constituent binary entropy terms.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from benchmark_mc_maze import BIN_SIZE_S, MIB, SEED, _load_counts, _profile_call, _sample_sets, _write_csv
from dthoi import estimate_entropy, information_measures, prepare_data


ANALYSIS_ESTIMATORS = (
    "empirical",
    "miller_madow",
    "schurmann",
    "shrinkage",
    "chao_shen",
    "pitman_yor",
)
BIN_SIZES_S = (0.005, 0.010, 0.020, 0.050)
CONVERGENCE_TARGETS = (512, 1024, 2048, 4096, 8192)
ORDERS = (5, 8)
N_SETS = 48
MEMORY_ORDERS = (3, 5, 8, 12, 15)
MEMORY_SETS = 128


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including exact ties, without SciPy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    """Compare two aligned vectors using finite values only."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    n = int(finite.sum())
    if n == 0:
        return {
            "finite_values": 0,
            "mae": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "sign_agreement": float("nan"),
        }
    x = left[finite]
    y = right[finite]
    mae = float(np.mean(np.abs(x - y)))
    pearson = float(np.corrcoef(x, y)[0, 1]) if n > 1 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
    xr = _rankdata(x)
    yr = _rankdata(y)
    spearman = (
        float(np.corrcoef(xr, yr)[0, 1])
        if n > 1 and np.std(xr) > 0 and np.std(yr) > 0
        else float("nan")
    )
    sign_agreement = float(np.mean(np.signbit(x) == np.signbit(y)))
    return {
        "finite_values": n,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "sign_agreement": sign_agreement,
    }


def _finite_mean(values: torch.Tensor) -> float:
    """Return the mean of finite tensor values."""
    flat = values.detach().cpu().to(torch.float64).reshape(-1)
    finite = flat[torch.isfinite(flat)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _finite_fraction_positive(values: torch.Tensor) -> float:
    """Return the fraction of finite tensor values strictly above zero."""
    flat = values.detach().cpu().to(torch.float64).reshape(-1)
    finite = flat[torch.isfinite(flat)]
    return float((finite > 0).to(torch.float64).mean()) if finite.numel() else float("nan")


def _make_common_sets(active_pool: np.ndarray) -> dict[int, torch.Tensor]:
    """Create deterministic variable sets shared by every stability analysis."""
    rng = np.random.default_rng(SEED + 101)
    return {
        order: _sample_sets(active_pool, order=order, count=N_SETS, rng=rng)
        for order in ORDERS
    }


def _dataset_at_20_ms(nwb_path: Path):
    """Load the canonical 20 ms binary representation and active-neuron pool."""
    counts = _load_counts(nwb_path, BIN_SIZE_S)
    binary = (counts > 0).astype(np.uint8, copy=False)
    duration_s = binary.shape[0] * BIN_SIZE_S
    rates_hz = counts.sum(axis=0) / duration_s
    active_pool = np.argsort(rates_hz)[-min(64, binary.shape[1]) :]
    return counts, binary, active_pool


def _run_convergence(
    binary: np.ndarray,
    prepared_full,
    sets_by_order: dict[int, torch.Tensor],
    output_dir: Path,
) -> None:
    """Measure nested-subsample convergence toward the full-data estimates."""
    n_samples = binary.shape[0]
    rng = np.random.default_rng(SEED + 202)
    permutation = rng.permutation(n_samples)
    sample_sizes = [n for n in CONVERGENCE_TARGETS if n < n_samples] + [n_samples]

    references: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for estimator in ANALYSIS_ESTIMATORS:
        for order, subsets in sets_by_order.items():
            entropy = estimate_entropy(prepared_full, subsets, entropy_estimator=estimator)
            measures = information_measures(prepared_full, subsets, entropy_estimator=estimator)
            references[(estimator, order)] = (
                entropy[:, 0].detach().cpu().numpy(),
                measures.o_information[:, 0].detach().cpu().numpy(),
            )

    rows: list[dict] = []
    for n in sample_sizes:
        prepared = prepared_full if n == n_samples else prepare_data(binary[permutation[:n]])
        for estimator in ANALYSIS_ESTIMATORS:
            for order, subsets in sets_by_order.items():
                entropy = estimate_entropy(prepared, subsets, entropy_estimator=estimator)
                measures = information_measures(prepared, subsets, entropy_estimator=estimator)
                entropy_values = entropy[:, 0].detach().cpu().numpy()
                o_values = measures.o_information[:, 0].detach().cpu().numpy()
                entropy_ref, o_ref = references[(estimator, order)]
                entropy_cmp = _comparison(entropy_values, entropy_ref)
                o_cmp = _comparison(o_values, o_ref)
                row = {
                    "estimator": estimator,
                    "order": order,
                    "samples": n,
                    "sample_fraction": n / n_samples,
                    "variable_sets": int(subsets.shape[0]),
                    "entropy_mean_bits": _finite_mean(entropy),
                    "entropy_mae_to_full_bits": entropy_cmp["mae"],
                    "entropy_spearman_to_full": entropy_cmp["spearman"],
                    "o_mean_bits": _finite_mean(measures.o_information),
                    "o_fraction_positive": _finite_fraction_positive(measures.o_information),
                    "o_mae_to_full_bits": o_cmp["mae"],
                    "o_pearson_to_full": o_cmp["pearson"],
                    "o_spearman_to_full": o_cmp["spearman"],
                    "o_sign_agreement_to_full": o_cmp["sign_agreement"],
                    "finite_o_values": o_cmp["finite_values"],
                }
                rows.append(row)
                print("CONVERGENCE", json.dumps(row, sort_keys=True))
    _write_csv(output_dir / "convergence_by_sample_size.csv", rows)


def _run_binning(
    nwb_path: Path,
    active_pool: np.ndarray,
    sets_by_order: dict[int, torch.Tensor],
    output_dir: Path,
) -> None:
    """Measure estimator sensitivity to 5, 10, 20, and 50 ms spike bins."""
    raw_rows: list[dict] = []
    vectors: dict[tuple[float, str, int], np.ndarray] = {}

    for bin_size_s in BIN_SIZES_S:
        counts = _load_counts(nwb_path, bin_size_s)
        binary = (counts > 0).astype(np.uint8, copy=False)
        prepared = prepare_data(binary)
        total_spikes = int(counts.sum())
        discarded = int((counts - binary).sum())
        for estimator in ANALYSIS_ESTIMATORS:
            for order, subsets in sets_by_order.items():
                entropy = estimate_entropy(prepared, subsets, entropy_estimator=estimator)
                measures = information_measures(prepared, subsets, entropy_estimator=estimator)
                o_values = measures.o_information[:, 0].detach().cpu().numpy()
                vectors[(bin_size_s, estimator, order)] = o_values
                raw_rows.append(
                    {
                        "bin_size_ms": 1000.0 * bin_size_s,
                        "samples": int(binary.shape[0]),
                        "estimator": estimator,
                        "order": order,
                        "variable_sets": int(subsets.shape[0]),
                        "binary_occupancy_mean": float(binary[:, active_pool].mean()),
                        "count_cells_gt1_fraction": float((counts[:, active_pool] > 1).mean()),
                        "spikes_lost_by_binarization_fraction": (
                            float(discarded / total_spikes) if total_spikes else 0.0
                        ),
                        "entropy_mean_bits": _finite_mean(entropy),
                        "o_mean_bits": _finite_mean(measures.o_information),
                        "o_fraction_positive": _finite_fraction_positive(measures.o_information),
                    }
                )

    rows: list[dict] = []
    for row in raw_rows:
        key = (row["bin_size_ms"] / 1000.0, row["estimator"], row["order"])
        ref_key = (0.020, row["estimator"], row["order"])
        cmp = _comparison(vectors[key], vectors[ref_key])
        row.update(
            {
                "o_mae_to_20ms_bits": cmp["mae"],
                "o_pearson_to_20ms": cmp["pearson"],
                "o_spearman_to_20ms": cmp["spearman"],
                "o_sign_agreement_to_20ms": cmp["sign_agreement"],
            }
        )
        rows.append(row)
        print("BINNING", json.dumps(row, sort_keys=True))
    _write_csv(output_dir / "binning_sensitivity.csv", rows)


def _run_split_half(
    binary: np.ndarray,
    sets_by_order: dict[int, torch.Tensor],
    output_dir: Path,
) -> None:
    """Measure sign and ranking stability between deterministic random halves."""
    rng = np.random.default_rng(SEED + 303)
    permutation = rng.permutation(binary.shape[0])
    midpoint = len(permutation) // 2
    left = prepare_data(binary[permutation[:midpoint]])
    right = prepare_data(binary[permutation[midpoint : 2 * midpoint]])

    rows: list[dict] = []
    for estimator in ANALYSIS_ESTIMATORS:
        for order, subsets in sets_by_order.items():
            left_result = information_measures(left, subsets, entropy_estimator=estimator)
            right_result = information_measures(right, subsets, entropy_estimator=estimator)
            left_o = left_result.o_information[:, 0].detach().cpu().numpy()
            right_o = right_result.o_information[:, 0].detach().cpu().numpy()
            cmp = _comparison(left_o, right_o)
            row = {
                "estimator": estimator,
                "order": order,
                "samples_per_half": midpoint,
                "variable_sets": int(subsets.shape[0]),
                "o_mean_half_a_bits": _finite_mean(left_result.o_information),
                "o_mean_half_b_bits": _finite_mean(right_result.o_information),
                "o_fraction_positive_half_a": _finite_fraction_positive(left_result.o_information),
                "o_fraction_positive_half_b": _finite_fraction_positive(right_result.o_information),
                "o_mae_between_halves_bits": cmp["mae"],
                "o_pearson_between_halves": cmp["pearson"],
                "o_spearman_between_halves": cmp["spearman"],
                "o_sign_agreement_between_halves": cmp["sign_agreement"],
                "finite_o_values": cmp["finite_values"],
            }
            rows.append(row)
            print("SPLIT_HALF", json.dumps(row, sort_keys=True))
    _write_csv(output_dir / "split_half_stability.csv", rows)


def _memory_worker(nwb_path: Path, order: int, variable_sets: int) -> None:
    """Run one isolated empirical global-measure memory case and print JSON."""
    torch.set_num_threads(1)
    counts, binary, active_pool = _dataset_at_20_ms(nwb_path)
    del counts
    prepared = prepare_data(binary)
    rng = np.random.default_rng(SEED + 404 + order)
    subsets = _sample_sets(active_pool, order=order, count=variable_sets, rng=rng)
    result, elapsed, peak_mem, caught = _profile_call(
        lambda: information_measures(prepared, subsets, entropy_estimator="empirical")
    )
    row = {
        "estimator": "empirical",
        "order": order,
        "variable_sets": variable_sets,
        "samples": int(binary.shape[0]),
        "seconds": elapsed,
        "isolated_peak_rss_growth_mib": peak_mem,
        "o_mean_bits": _finite_mean(result.o_information),
        "warning_count": len(caught),
    }
    print("MEMORY_WORKER", json.dumps(row, sort_keys=True))


def _run_isolated_memory(nwb_path: Path, output_dir: Path) -> None:
    """Launch each memory case in a fresh Python process."""
    rows: list[dict] = []
    script = Path(__file__).resolve()
    for order in MEMORY_ORDERS:
        command = [
            sys.executable,
            str(script),
            "--nwb-path",
            str(nwb_path),
            "--memory-worker",
            "--memory-order",
            str(order),
            "--memory-sets",
            str(MEMORY_SETS),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        payload = None
        for line in completed.stdout.splitlines():
            if line.startswith("MEMORY_WORKER "):
                payload = json.loads(line.split(" ", 1)[1])
        if payload is None:
            raise RuntimeError(
                f"Memory worker for order {order} produced no parseable result.\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        rows.append(payload)
        print("ISOLATED_MEMORY", json.dumps(payload, sort_keys=True))
    _write_csv(output_dir / "isolated_memory_scaling.csv", rows)


def run(nwb_path: Path, output_dir: Path) -> None:
    """Run all real-data stability and isolated-memory analyses."""
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, binary, active_pool = _dataset_at_20_ms(nwb_path)
    prepared_full = prepare_data(binary)
    sets_by_order = _make_common_sets(active_pool)
    _run_convergence(binary, prepared_full, sets_by_order, output_dir)
    _run_binning(nwb_path, active_pool, sets_by_order, output_dir)
    _run_split_half(binary, sets_by_order, output_dir)
    _run_isolated_memory(nwb_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("benchmark-results"), type=Path)
    parser.add_argument("--memory-worker", action="store_true")
    parser.add_argument("--memory-order", type=int)
    parser.add_argument("--memory-sets", type=int, default=MEMORY_SETS)
    args = parser.parse_args()
    if args.memory_worker:
        if args.memory_order is None:
            raise SystemExit("--memory-order is required with --memory-worker")
        _memory_worker(args.nwb_path, args.memory_order, args.memory_sets)
    else:
        run(args.nwb_path, args.output_dir)
