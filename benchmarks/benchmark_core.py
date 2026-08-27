"""Reproducible CPU baselines for the dTHOI discrete core.

Run from an environment with dTHOI installed in editable mode:

    python benchmarks/benchmark_core.py

This is a performance benchmark, not a correctness test. It intentionally uses
one execution device and never compares execution devices.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import time

import torch

from dthoi.entropy.base import CountingEntropyProvider
from dthoi.entropy.counting import (
    dense_counts_from_codes,
    encode_binary_rows,
    encode_binary_states,
    mask_binary_rows,
    sparse_counts_from_codes,
)
from dthoi.measures.core import nplets_measures
from dthoi.subsets import masks_from_subsets


CASES = (
    (1000, 32, 5, 128),
    (1000, 60, 10, 128),
    (1000, 60, 15, 128),
    (5000, 60, 15, 128),
    (2000, 128, 15, 128),
    (1000, 60, 8, 512),
)


def _subset_batch(n_variables: int, order: int, batch_size: int) -> torch.Tensor:
    rows = list(
        itertools.islice(
            itertools.combinations(range(n_variables), order),
            batch_size,
        )
    )
    return torch.tensor(rows, dtype=torch.long)


def _median_seconds(fn, repeats: int) -> float:
    fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def _emit(stage: str, T: int, N: int, K: int, B: int, seconds: float) -> None:
    print(f"{stage},{T},{N},{K},{B},{1000.0 * seconds:.3f}")


def run(repeats: int) -> None:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20260827)

    print("stage,T,N,K,B,median_ms")
    for T, N, K, B in CASES:
        X = torch.randint(0, 2, (T, N), dtype=torch.uint8, generator=generator)
        subsets = _subset_batch(N, K, B)

        codes = encode_binary_states(X, subsets)
        _emit(
            "compressed_state_encoding",
            T,
            N,
            K,
            B,
            _median_seconds(lambda: encode_binary_states(X, subsets), repeats),
        )

        if N <= 63:
            row_codes = encode_binary_rows(X)
            masks = masks_from_subsets(subsets)
            _emit(
                "row_state_encoding",
                T,
                N,
                K,
                B,
                _median_seconds(lambda: encode_binary_rows(X), repeats),
            )
            _emit(
                "sparse_mask_encoding",
                T,
                N,
                K,
                B,
                _median_seconds(lambda: mask_binary_rows(row_codes, masks), repeats),
            )

        if (1 << K) <= (1 << 16):
            _emit(
                "dense_counting",
                T,
                N,
                K,
                B,
                _median_seconds(
                    lambda: dense_counts_from_codes(codes, 1 << K),
                    repeats,
                ),
            )

        _emit(
            "sparse_counting",
            T,
            N,
            K,
            B,
            _median_seconds(lambda: sparse_counts_from_codes(codes), repeats),
        )
        _emit(
            "entropy_provider",
            T,
            N,
            K,
            B,
            _median_seconds(
                lambda: CountingEntropyProvider(X, count_mode="auto").entropy(subsets),
                repeats,
            ),
        )
        _emit(
            "nplets_measures",
            T,
            N,
            K,
            B,
            _median_seconds(
                lambda: nplets_measures(X, subsets, count_mode="auto"),
                repeats,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    run(args.repeats)
