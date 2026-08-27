from __future__ import annotations

import itertools
import math
import warnings

import torch

import dthoi
from dthoi.entropy.base import CountingEntropyProvider


NOISE = torch.tensor([0.34, 0.36, 0.38, 0.40, 0.42, 0.44], dtype=torch.float64)
N_VARIABLES = int(NOISE.numel())
FULL_SET = torch.arange(N_VARIABLES, dtype=torch.long)
SAMPLE_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
N_REPLICATES = 32
SEED = 20260827

GLOBAL_ESTIMATORS = (
    "empirical",
    "miller_madow",
    "schurmann",
    "shrinkage",
    "chao_shen",
    "pitman_yor",
    "ansb",
)
LOCAL_ESTIMATORS = (
    "empirical",
    "miller_madow",
    "schurmann",
    "chao_shen",
)
METRICS = ("H", "TC", "DTC", "O", "S")


def subset_probabilities(subset: torch.Tensor) -> torch.Tensor:
    """Return the exact state probabilities for one variable subset."""
    subset = subset.to(dtype=torch.long)
    order = int(subset.numel())
    states = torch.arange(1 << order, dtype=torch.long)
    bits = ((states[:, None] >> torch.arange(order)) & 1).to(torch.float64)
    q = NOISE[subset]
    given_zero = torch.where(bits.bool(), q, 1.0 - q).prod(dim=1)
    given_one = torch.where(bits.bool(), 1.0 - q, q).prod(dim=1)
    return 0.5 * (given_zero + given_one)


def exact_entropy(subset: torch.Tensor) -> torch.Tensor:
    """Calculate exact Shannon entropy in bits from the known model."""
    probabilities = subset_probabilities(subset)
    return -(probabilities * torch.log2(probabilities)).sum()


def exact_local_entropy(data: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    """Calculate exact pointwise surprisal for sampled observations."""
    subset = subset.to(dtype=torch.long)
    probabilities = subset_probabilities(subset)
    weights = 1 << torch.arange(subset.numel(), dtype=torch.long)
    codes = (data[:, subset].to(torch.long) * weights).sum(dim=1)
    return -torch.log2(probabilities[codes])


def exact_global_metrics() -> dict[str, float]:
    """Return exact entropy and four information measures for the full set."""
    h_joint = exact_entropy(FULL_SET)
    h_single = torch.stack([exact_entropy(torch.tensor([i])) for i in range(N_VARIABLES)]).sum()
    h_loo = torch.stack(
        [
            exact_entropy(FULL_SET[FULL_SET != i])
            for i in range(N_VARIABLES)
        ]
    ).sum()
    tc = h_single - h_joint
    dtc = h_loo - (N_VARIABLES - 1) * h_joint
    return {
        "H": float(h_joint),
        "TC": float(tc),
        "DTC": float(dtc),
        "O": float(tc - dtc),
        "S": float(tc + dtc),
    }


def exact_local_metrics(data: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return exact sample-resolved entropy and information measures."""
    h_joint = exact_local_entropy(data, FULL_SET)
    h_single = torch.stack(
        [exact_local_entropy(data, torch.tensor([i])) for i in range(N_VARIABLES)]
    ).sum(dim=0)
    h_loo = torch.stack(
        [
            exact_local_entropy(data, FULL_SET[FULL_SET != i])
            for i in range(N_VARIABLES)
        ]
    ).sum(dim=0)
    tc = h_single - h_joint
    dtc = h_loo - (N_VARIABLES - 1) * h_joint
    return {
        "H": h_joint,
        "TC": tc,
        "DTC": dtc,
        "O": tc - dtc,
        "S": tc + dtc,
    }


def sample_model(sample_count: int, generator: torch.Generator) -> torch.Tensor:
    """Draw exact samples from the latent-common-cause binary model."""
    probabilities = subset_probabilities(FULL_SET)
    states = torch.multinomial(
        probabilities,
        sample_count,
        replacement=True,
        generator=generator,
    )
    return ((states[:, None] >> torch.arange(N_VARIABLES)) & 1).to(torch.uint8)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return finite-sample Pearson correlation, ignoring non-finite pairs."""
    x = x.to(torch.float64).flatten()
    y = y.to(torch.float64).flatten()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if denominator == 0:
        return float("nan")
    return float((x * y).sum() / denominator)


def rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return RMSE over finite paired values."""
    x = x.to(torch.float64).flatten()
    y = y.to(torch.float64).flatten()
    valid = torch.isfinite(x) & torch.isfinite(y)
    if not bool(valid.any()):
        return float("nan")
    return float(torch.sqrt(torch.mean((x[valid] - y[valid]).square())))


def finite_mean(values: list[float]) -> float:
    """Return arithmetic mean after removing NaN and infinities."""
    tensor = torch.tensor(values, dtype=torch.float64)
    tensor = tensor[torch.isfinite(tensor)]
    return float(tensor.mean()) if tensor.numel() else float("nan")


def finite_sd(values: list[float]) -> float:
    """Return sample standard deviation after removing non-finite values."""
    tensor = torch.tensor(values, dtype=torch.float64)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() < 2:
        return 0.0 if tensor.numel() == 1 else float("nan")
    return float(tensor.std(unbiased=True))


def estimator_global(data: torch.Tensor, estimator: str) -> dict[str, float]:
    """Calculate repo entropy and information estimates for the full variable set."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h = dthoi.estimate_entropy(
            data,
            FULL_SET.unsqueeze(0),
            entropy_estimator=estimator,
        )[0, 0]
        measures = dthoi.information_measures(
            data,
            FULL_SET.unsqueeze(0),
            entropy_estimator=estimator,
        ).values[0, 0]
    return {
        "H": float(h),
        "TC": float(measures[0]),
        "DTC": float(measures[1]),
        "O": float(measures[2]),
        "S": float(measures[3]),
    }


def estimator_local(data: torch.Tensor, estimator: str) -> dict[str, torch.Tensor]:
    """Calculate repo local entropy and local information values for one estimator."""
    provider = CountingEntropyProvider(data, estimator=estimator, count_mode="auto")
    _, local_entropy = provider.entropy_with_local(FULL_SET.unsqueeze(0))
    result = dthoi.information_measures(
        data,
        FULL_SET.unsqueeze(0),
        entropy_estimator=estimator,
        local_values=True,
    )
    local_measures = result.local.values[0][0]
    return {
        "H": local_entropy[0][0],
        "TC": local_measures[:, 0],
        "DTC": local_measures[:, 1],
        "O": local_measures[:, 2],
        "S": local_measures[:, 3],
    }


def print_matrix(title: str, columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    """Print a compact CSV-like validation table."""
    print(f"\n[{title}]")
    print(",".join(("N", "metric", *columns)))
    for row in rows:
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append("nan" if not math.isfinite(value) else f"{value:.8f}")
            else:
                formatted.append(str(value))
        print(",".join(formatted))


def main() -> None:
    """Run reproducible weak-correlation ground-truth validation."""
    exact_global = exact_global_metrics()
    signals = 1.0 - 2.0 * NOISE
    pairwise_correlations = torch.tensor(
        [
            float(signals[i] * signals[j])
            for i, j in itertools.combinations(range(N_VARIABLES), 2)
        ],
        dtype=torch.float64,
    )

    print("[MODEL]")
    print(f"variables={N_VARIABLES}")
    print("noise=" + ",".join(f"{float(q):.2f}" for q in NOISE))
    print(f"pairwise_corr_min={float(pairwise_correlations.min()):.8f}")
    print(f"pairwise_corr_median={float(pairwise_correlations.median()):.8f}")
    print(f"pairwise_corr_max={float(pairwise_correlations.max()):.8f}")
    for metric in METRICS:
        print(f"ground_truth_{metric}={exact_global[metric]:.12f}")

    global_errors: dict[tuple[int, str, str], list[float]] = {}
    local_truth_corr: dict[tuple[int, str, str], list[float]] = {}
    local_truth_rmse: dict[tuple[int, str, str], list[float]] = {}
    local_pair_corr: dict[tuple[int, str, str, str], list[float]] = {}

    generator = torch.Generator().manual_seed(SEED)
    max_sample_size = max(SAMPLE_SIZES)

    for replicate in range(N_REPLICATES):
        full_data = sample_model(max_sample_size, generator)
        for sample_size in SAMPLE_SIZES:
            data = full_data[:sample_size]
            truth_local = exact_local_metrics(data)

            for estimator in GLOBAL_ESTIMATORS:
                estimate = estimator_global(data, estimator)
                for metric in METRICS:
                    global_errors.setdefault((sample_size, estimator, metric), []).append(
                        estimate[metric] - exact_global[metric]
                    )

            local_estimates: dict[str, dict[str, torch.Tensor]] = {}
            for estimator in LOCAL_ESTIMATORS:
                local = estimator_local(data, estimator)
                local_estimates[estimator] = local
                for metric in METRICS:
                    local_truth_corr.setdefault((sample_size, estimator, metric), []).append(
                        pearson(local[metric], truth_local[metric])
                    )
                    local_truth_rmse.setdefault((sample_size, estimator, metric), []).append(
                        rmse(local[metric], truth_local[metric])
                    )

            for first, second in itertools.combinations(LOCAL_ESTIMATORS, 2):
                for metric in METRICS:
                    local_pair_corr.setdefault(
                        (sample_size, metric, first, second), []
                    ).append(
                        pearson(
                            local_estimates[first][metric],
                            local_estimates[second][metric],
                        )
                    )

        print(f"completed_replicate={replicate + 1}/{N_REPLICATES}")

    global_bias_rows: list[tuple[object, ...]] = []
    global_rmse_rows: list[tuple[object, ...]] = []
    for sample_size in SAMPLE_SIZES:
        for metric in METRICS:
            biases = []
            rmses = []
            for estimator in GLOBAL_ESTIMATORS:
                errors = global_errors[(sample_size, estimator, metric)]
                biases.append(finite_mean(errors))
                rmses.append(math.sqrt(finite_mean([value * value for value in errors])))
            global_bias_rows.append((sample_size, metric, *biases))
            global_rmse_rows.append((sample_size, metric, *rmses))

    print_matrix("GLOBAL_BIAS_BITS", GLOBAL_ESTIMATORS, global_bias_rows)
    print_matrix("GLOBAL_RMSE_BITS", GLOBAL_ESTIMATORS, global_rmse_rows)

    local_truth_corr_rows: list[tuple[object, ...]] = []
    local_truth_rmse_rows: list[tuple[object, ...]] = []
    for sample_size in SAMPLE_SIZES:
        for metric in METRICS:
            local_truth_corr_rows.append(
                (
                    sample_size,
                    metric,
                    *[
                        finite_mean(local_truth_corr[(sample_size, estimator, metric)])
                        for estimator in LOCAL_ESTIMATORS
                    ],
                )
            )
            local_truth_rmse_rows.append(
                (
                    sample_size,
                    metric,
                    *[
                        finite_mean(local_truth_rmse[(sample_size, estimator, metric)])
                        for estimator in LOCAL_ESTIMATORS
                    ],
                )
            )

    print_matrix("LOCAL_VS_TRUTH_CORR", LOCAL_ESTIMATORS, local_truth_corr_rows)
    print_matrix("LOCAL_VS_TRUTH_RMSE_BITS", LOCAL_ESTIMATORS, local_truth_rmse_rows)

    pairs = tuple(f"{first}|{second}" for first, second in itertools.combinations(LOCAL_ESTIMATORS, 2))
    pair_rows: list[tuple[object, ...]] = []
    pair_sd_rows: list[tuple[object, ...]] = []
    for sample_size in SAMPLE_SIZES:
        for metric in METRICS:
            means = []
            sds = []
            for first, second in itertools.combinations(LOCAL_ESTIMATORS, 2):
                values = local_pair_corr[(sample_size, metric, first, second)]
                means.append(finite_mean(values))
                sds.append(finite_sd(values))
            pair_rows.append((sample_size, metric, *means))
            pair_sd_rows.append((sample_size, metric, *sds))

    print_matrix("LOCAL_PAIRWISE_CORR", pairs, pair_rows)
    print_matrix("LOCAL_PAIRWISE_CORR_SD", pairs, pair_sd_rows)

    print("\n[VALIDATION_CONTRACTS]")
    print("global_estimators=" + ",".join(GLOBAL_ESTIMATORS))
    print("local_estimators=" + ",".join(LOCAL_ESTIMATORS))
    print("unsupported_local_estimators=shrinkage,pitman_yor,ansb")
    print(f"replicates={N_REPLICATES}")
    print("sample_sizes=" + ",".join(str(n) for n in SAMPLE_SIZES))


if __name__ == "__main__":
    main()
