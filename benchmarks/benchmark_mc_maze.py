"""Benchmark dTHOI on the public NLB MC_Maze_Small spike dataset.

This script is an exploratory real-data benchmark, not a correctness test. It
loads spike-sorted macaque M1/PMd activity from NWB, bins spikes at 20 ms, and
binarizes each neuron/bin as required by dTHOI's current binary data contract.

The benchmark records estimator outputs, wall time, process RSS growth, local
value decomposition checks, and basic scaling on one CPU execution device.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import psutil
import torch
from pynwb import NWBHDF5IO

import dthoi
from dthoi import estimate_entropy, information_measures, prepare_data


ESTIMATORS = (
    "empirical",
    "miller_madow",
    "schurmann",
    "shrinkage",
    "chao_shen",
    "pitman_yor",
    "ansb",
)
LOCAL_ESTIMATORS = ("empirical", "miller_madow", "schurmann", "chao_shen")
BIN_SIZE_S = 0.020
SEED = 20260828
MIB = 1024.0**2


def _load_counts(path: Path, bin_size_s: float) -> np.ndarray:
    """Load all sorted units and bin spike times over the behavior time span."""
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        units = nwb.units
        spike_times = [np.asarray(units["spike_times"][i]) for i in range(len(units))]
        hand = nwb.processing["behavior"].data_interfaces["hand_vel"]
        behavior_timestamps = np.asarray(hand.timestamps[:])

    t0 = float(behavior_timestamps[0])
    t1 = float(behavior_timestamps[-1])
    edges = np.arange(t0, t1, bin_size_s)
    if len(edges) < 2:
        raise RuntimeError("Behavior interval is too short for the requested bin size.")
    return np.stack(
        [np.histogram(times, bins=edges)[0] for times in spike_times], axis=1
    ).astype(np.int16, copy=False)


def _sample_sets(pool: np.ndarray, order: int, count: int, rng: np.random.Generator) -> torch.Tensor:
    """Draw deterministic unique variable sets from a supplied neuron pool."""
    rows: set[tuple[int, ...]] = set()
    while len(rows) < count:
        rows.add(tuple(sorted(int(x) for x in rng.choice(pool, size=order, replace=False))))
    return torch.tensor(sorted(rows), dtype=torch.long)


def _profile_call(fn):
    """Return output, elapsed seconds, peak RSS growth MiB, and captured warnings."""
    gc.collect()
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def poll() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, process.memory_info().rss)
            time.sleep(0.002)
        peak = max(peak, process.memory_info().rss)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    captured = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = fn()
            captured = [str(item.message) for item in caught]
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join()
    return output, elapsed, max(0.0, (peak - baseline) / MIB), captured


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of homogeneous dictionaries to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_summary(values: torch.Tensor) -> tuple[float, float, float]:
    flat = values.detach().cpu().to(torch.float64).reshape(-1)
    finite = torch.isfinite(flat)
    nan_fraction = 1.0 - float(finite.to(torch.float64).mean())
    if not bool(finite.any()):
        return float("nan"), float("nan"), nan_fraction
    kept = flat[finite]
    return float(kept.mean()), float(kept.std(unbiased=False)), nan_fraction


def run(nwb_path: Path, output_dir: Path) -> None:
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    counts, load_s, load_mem, load_warnings = _profile_call(
        lambda: _load_counts(nwb_path, BIN_SIZE_S)
    )
    binary = (counts > 0).astype(np.uint8, copy=False)
    n_samples, n_units = binary.shape
    duration_s = n_samples * BIN_SIZE_S
    rates_hz = counts.sum(axis=0) / duration_s
    occupancies = binary.mean(axis=0)
    active_pool = np.argsort(rates_hz)[-min(64, n_units) :]

    total_spikes = int(counts.sum())
    discarded = int((counts - binary).sum())
    summary = {
        "dataset": "NLB MC_Maze_Small",
        "dandi": "000140",
        "dandi_version": "0.220113.0408",
        "source_file": nwb_path.name,
        "bin_size_ms": BIN_SIZE_S * 1000.0,
        "samples": int(n_samples),
        "units": int(n_units),
        "duration_s": float(duration_s),
        "binary_occupancy_mean": float(binary.mean()),
        "binary_occupancy_median_unit": float(np.median(occupancies)),
        "firing_rate_hz_median": float(np.median(rates_hz)),
        "firing_rate_hz_min": float(rates_hz.min()),
        "firing_rate_hz_max": float(rates_hz.max()),
        "count_cells_gt1_fraction": float((counts > 1).mean()),
        "max_spikes_in_bin": int(counts.max()),
        "spikes_lost_by_binarization_fraction": float(discarded / total_spikes) if total_spikes else 0.0,
        "active_pool_size": int(len(active_pool)),
        "active_pool_rate_hz_min": float(rates_hz[active_pool].min()),
        "load_seconds": load_s,
        "load_peak_rss_growth_mib": load_mem,
        "load_warning_count": len(load_warnings),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "dthoi_version": getattr(dthoi, "__version__", "unversioned"),
        "torch_threads": torch.get_num_threads(),
        "seed": SEED,
    }

    prepared, prep_s, prep_mem, prep_warnings = _profile_call(lambda: prepare_data(binary))
    summary.update(
        {
            "prepare_data_seconds": prep_s,
            "prepare_data_peak_rss_growth_mib": prep_mem,
            "prepare_data_warning_count": len(prep_warnings),
        }
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("DATASET_SUMMARY", json.dumps(summary, sort_keys=True))

    entropy_sets = _sample_sets(active_pool, order=8, count=128, rng=rng)
    entropy_rows = []
    for estimator in ESTIMATORS:
        try:
            result, elapsed, peak_mem, caught = _profile_call(
                lambda estimator=estimator: estimate_entropy(
                    prepared, entropy_sets, entropy_estimator=estimator
                )
            )
            mean, std, nan_fraction = _finite_summary(result)
            row = {
                "estimator": estimator,
                "order": 8,
                "variable_sets": int(entropy_sets.shape[0]),
                "seconds": elapsed,
                "peak_rss_growth_mib": peak_mem,
                "entropy_mean_bits": mean,
                "entropy_std_bits": std,
                "nan_fraction": nan_fraction,
                "warning_count": len(caught),
                "warning_sample": " | ".join(caught[:3]),
                "status": "ok",
            }
        except Exception as exc:  # exploratory benchmark should record failures
            row = {
                "estimator": estimator,
                "order": 8,
                "variable_sets": int(entropy_sets.shape[0]),
                "seconds": float("nan"),
                "peak_rss_growth_mib": float("nan"),
                "entropy_mean_bits": float("nan"),
                "entropy_std_bits": float("nan"),
                "nan_fraction": float("nan"),
                "warning_count": 0,
                "warning_sample": "",
                "status": f"error: {type(exc).__name__}: {exc}",
            }
        entropy_rows.append(row)
        print("ENTROPY", json.dumps(row, sort_keys=True, default=str))
    _write_csv(output_dir / "entropy_estimators.csv", entropy_rows)

    measure_rows = []
    for order in (5, 8):
        subsets = _sample_sets(active_pool, order=order, count=64, rng=rng)
        for estimator in ESTIMATORS:
            try:
                result, elapsed, peak_mem, caught = _profile_call(
                    lambda estimator=estimator, subsets=subsets: information_measures(
                        prepared, subsets, entropy_estimator=estimator
                    )
                )
                o_mean, o_std, nan_fraction = _finite_summary(result.o_information)
                tc_mean, _, _ = _finite_summary(result.total_correlation)
                dtc_mean, _, _ = _finite_summary(result.dual_total_correlation)
                o_err = float(
                    torch.nan_to_num(
                        (result.o_information - (result.total_correlation - result.dual_total_correlation)).abs(),
                        nan=0.0,
                    ).max()
                )
                s_err = float(
                    torch.nan_to_num(
                        (result.s_information - (result.total_correlation + result.dual_total_correlation)).abs(),
                        nan=0.0,
                    ).max()
                )
                row = {
                    "estimator": estimator,
                    "order": order,
                    "variable_sets": int(subsets.shape[0]),
                    "seconds": elapsed,
                    "peak_rss_growth_mib": peak_mem,
                    "tc_mean_bits": tc_mean,
                    "dtc_mean_bits": dtc_mean,
                    "o_mean_bits": o_mean,
                    "o_std_bits": o_std,
                    "nan_fraction": nan_fraction,
                    "o_identity_max_abs_error": o_err,
                    "s_identity_max_abs_error": s_err,
                    "warning_count": len(caught),
                    "warning_sample": " | ".join(caught[:3]),
                    "status": "ok",
                }
            except Exception as exc:
                row = {
                    "estimator": estimator,
                    "order": order,
                    "variable_sets": int(subsets.shape[0]),
                    "seconds": float("nan"),
                    "peak_rss_growth_mib": float("nan"),
                    "tc_mean_bits": float("nan"),
                    "dtc_mean_bits": float("nan"),
                    "o_mean_bits": float("nan"),
                    "o_std_bits": float("nan"),
                    "nan_fraction": float("nan"),
                    "o_identity_max_abs_error": float("nan"),
                    "s_identity_max_abs_error": float("nan"),
                    "warning_count": 0,
                    "warning_sample": "",
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
            measure_rows.append(row)
            print("MEASURES", json.dumps(row, sort_keys=True, default=str))
    _write_csv(output_dir / "information_measures.csv", measure_rows)

    local_sets = _sample_sets(active_pool, order=5, count=16, rng=rng)
    local_rows = []
    local_o: dict[str, np.ndarray] = {}
    for estimator in LOCAL_ESTIMATORS:
        try:
            result, elapsed, peak_mem, caught = _profile_call(
                lambda estimator=estimator: information_measures(
                    prepared,
                    local_sets,
                    entropy_estimator=estimator,
                    local_values=True,
                    max_local_values_per_batch=250_000,
                )
            )
            assert result.local is not None
            local_values = result.local.values[0]
            local_o_tensor = result.local.o_information[0]
            local_o[estimator] = local_o_tensor.detach().cpu().numpy().reshape(-1)
            global_from_local = local_values.mean(dim=1)
            mean_recovery_error = float(
                torch.nan_to_num((global_from_local - result.values[:, 0, :]).abs(), nan=0.0).max()
            )
            local_o_identity_error = float(
                torch.nan_to_num(
                    (
                        result.local.o_information[0]
                        - (result.local.total_correlation[0] - result.local.dual_total_correlation[0])
                    ).abs(),
                    nan=0.0,
                ).max()
            )
            o_flat = local_o_tensor.detach().cpu().to(torch.float64).reshape(-1)
            finite = o_flat[torch.isfinite(o_flat)]
            quantiles = (
                torch.quantile(finite, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
                if finite.numel()
                else torch.full((3,), float("nan"), dtype=torch.float64)
            )
            row = {
                "estimator": estimator,
                "order": 5,
                "variable_sets": int(local_sets.shape[0]),
                "samples": int(n_samples),
                "seconds": elapsed,
                "peak_rss_growth_mib": peak_mem,
                "returned_local_mib": local_values.nelement() * local_values.element_size() / MIB,
                "global_recovery_max_abs_error": mean_recovery_error,
                "local_o_identity_max_abs_error": local_o_identity_error,
                "local_o_mean": float(finite.mean()) if finite.numel() else float("nan"),
                "local_o_std": float(finite.std(unbiased=False)) if finite.numel() else float("nan"),
                "local_o_q01": float(quantiles[0]),
                "local_o_q50": float(quantiles[1]),
                "local_o_q99": float(quantiles[2]),
                "nan_fraction": 1.0 - float(torch.isfinite(o_flat).to(torch.float64).mean()),
                "warning_count": len(caught),
                "warning_sample": " | ".join(caught[:3]),
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "estimator": estimator,
                "order": 5,
                "variable_sets": int(local_sets.shape[0]),
                "samples": int(n_samples),
                "seconds": float("nan"),
                "peak_rss_growth_mib": float("nan"),
                "returned_local_mib": float("nan"),
                "global_recovery_max_abs_error": float("nan"),
                "local_o_identity_max_abs_error": float("nan"),
                "local_o_mean": float("nan"),
                "local_o_std": float("nan"),
                "local_o_q01": float("nan"),
                "local_o_q50": float("nan"),
                "local_o_q99": float("nan"),
                "nan_fraction": float("nan"),
                "warning_count": 0,
                "warning_sample": "",
                "status": f"error: {type(exc).__name__}: {exc}",
            }
        local_rows.append(row)
        print("LOCAL", json.dumps(row, sort_keys=True, default=str))
    _write_csv(output_dir / "local_information.csv", local_rows)

    corr_rows = []
    names = list(local_o)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            x = local_o[left]
            y = local_o[right]
            finite = np.isfinite(x) & np.isfinite(y)
            corr = float(np.corrcoef(x[finite], y[finite])[0, 1]) if finite.sum() > 1 else float("nan")
            corr_rows.append(
                {
                    "estimator_a": left,
                    "estimator_b": right,
                    "pearson_local_o": corr,
                    "finite_values": int(finite.sum()),
                }
            )
    _write_csv(output_dir / "local_o_correlations.csv", corr_rows)
    for row in corr_rows:
        print("LOCAL_CORR", json.dumps(row, sort_keys=True))

    scaling_rows = []
    for order in (3, 5, 8, 12):
        subsets = _sample_sets(active_pool, order=order, count=128, rng=rng)
        times = []
        memories = []
        for _ in range(3):
            _, elapsed, peak_mem, caught = _profile_call(
                lambda subsets=subsets: information_measures(
                    prepared, subsets, entropy_estimator="empirical"
                )
            )
            times.append(elapsed)
            memories.append(peak_mem)
            if caught:
                print("SCALING_WARNING", order, caught[:3])
        row = {
            "estimator": "empirical",
            "order": order,
            "variable_sets": int(subsets.shape[0]),
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
            "max_peak_rss_growth_mib": max(memories),
        }
        scaling_rows.append(row)
        print("SCALING", json.dumps(row, sort_keys=True))
    _write_csv(output_dir / "empirical_scaling.csv", scaling_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("benchmark-results"), type=Path)
    args = parser.parse_args()
    run(args.nwb_path, args.output_dir)
