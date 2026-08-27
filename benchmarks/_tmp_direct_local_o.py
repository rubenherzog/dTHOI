from __future__ import annotations

import itertools
import math
import warnings

import torch

import dthoi


NOISE = torch.tensor([0.34, 0.36, 0.38, 0.40, 0.42, 0.44], dtype=torch.float64)
N_VARIABLES = int(NOISE.numel())
Q = 1 << N_VARIABLES
FULL_SET = torch.arange(N_VARIABLES, dtype=torch.long)
SAMPLE_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
N_REPLICATES = 32
SEED = 20260828


def support_bits() -> torch.Tensor:
    """Return all binary states with shape [Q, variables]."""
    states = torch.arange(Q, dtype=torch.long)
    return ((states[:, None] >> torch.arange(N_VARIABLES)) & 1).to(torch.uint8)


def exact_joint_probabilities() -> torch.Tensor:
    """Return exact full-state probabilities for the latent common-cause model."""
    bits = support_bits().to(torch.float64)
    given_zero = torch.where(bits.bool(), NOISE, 1.0 - NOISE).prod(dim=1)
    given_one = torch.where(bits.bool(), 1.0 - NOISE, NOISE).prod(dim=1)
    return 0.5 * (given_zero + given_one)


def codes_from_data(data: torch.Tensor, variables: torch.Tensor) -> torch.Tensor:
    """Encode selected binary variables into compact integer state codes."""
    variables = variables.to(torch.long)
    weights = 1 << torch.arange(variables.numel(), dtype=torch.long)
    return (data[:, variables].to(torch.long) * weights).sum(dim=1)


def marginalize(full_probabilities: torch.Tensor, variables: torch.Tensor) -> torch.Tensor:
    """Marginalize a full joint distribution onto selected variables."""
    variables = variables.to(torch.long)
    bits = support_bits()
    codes = codes_from_data(bits, variables)
    result = torch.zeros(1 << variables.numel(), dtype=torch.float64)
    result.scatter_add_(0, codes, full_probabilities)
    return result


def direct_local_o(full_probabilities: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """Evaluate local O-information directly from one coherent joint distribution."""
    full_codes = codes_from_data(data, FULL_SET)
    local_joint = -torch.log2(full_probabilities[full_codes])

    local_single_sum = torch.zeros(data.shape[0], dtype=torch.float64)
    local_loo_sum = torch.zeros(data.shape[0], dtype=torch.float64)
    for index in range(N_VARIABLES):
        singleton = torch.tensor([index], dtype=torch.long)
        singleton_probabilities = marginalize(full_probabilities, singleton)
        singleton_codes = codes_from_data(data, singleton)
        local_single_sum += -torch.log2(singleton_probabilities[singleton_codes])

        keep = FULL_SET[FULL_SET != index]
        loo_probabilities = marginalize(full_probabilities, keep)
        loo_codes = codes_from_data(data, keep)
        local_loo_sum += -torch.log2(loo_probabilities[loo_codes])

    return (N_VARIABLES - 2) * local_joint + local_single_sum - local_loo_sum


def shrink_full_joint(data: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Apply the repo James-Stein rule once to the complete Q-state distribution."""
    codes = codes_from_data(data, FULL_SET)
    counts = torch.bincount(codes, minlength=Q).to(torch.float64)
    total = float(data.shape[0])
    empirical = counts / total
    target = 1.0 / float(Q)
    sum_squared = empirical.square().sum()
    denominator = (empirical - target).square().sum()
    if total <= 1.0 or float(denominator) <= 0.0:
        shrinkage = 1.0
    else:
        numerator = (1.0 - sum_squared) / (total - 1.0)
        shrinkage = float((numerator / denominator).clamp(0.0, 1.0))
    probabilities = shrinkage * target + (1.0 - shrinkage) * empirical
    return probabilities, shrinkage


def empirical_full_joint(data: torch.Tensor) -> torch.Tensor:
    """Return the empirical Q-state joint distribution."""
    codes = codes_from_data(data, FULL_SET)
    return torch.bincount(codes, minlength=Q).to(torch.float64) / float(data.shape[0])


def exact_local_o(data: torch.Tensor) -> torch.Tensor:
    """Return ground-truth local O-information for sampled observations."""
    return direct_local_o(exact_joint_probabilities(), data)


def current_empirical_local_o(data: torch.Tensor) -> torch.Tensor:
    """Return local O-information from the current dTHOI empirical path."""
    result = dthoi.information_measures(
        data,
        FULL_SET.unsqueeze(0),
        entropy_estimator="empirical",
        local_values=True,
    )
    assert result.local is not None
    return result.local.values[0][0][:, 2]


def current_global_o(data: torch.Tensor, estimator: str) -> float:
    """Return current dTHOI global O-information for one estimator."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = dthoi.information_measures(
            data,
            FULL_SET.unsqueeze(0),
            entropy_estimator=estimator,
        )
    return float(result.values[0, 0, 2])


def entropy(probabilities: torch.Tensor) -> float:
    """Return Shannon entropy in bits for a normalized probability vector."""
    positive = probabilities > 0
    p = probabilities[positive]
    return float(-(p * torch.log2(p)).sum())


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return Pearson correlation over finite paired values."""
    x = x.to(torch.float64).flatten()
    y = y.to(torch.float64).flatten()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if denominator == 0:
        return float("nan")
    return float((x * y).sum() / denominator)


def rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return RMSE in bits over finite paired values."""
    valid = torch.isfinite(x) & torch.isfinite(y)
    return float(torch.sqrt((x[valid] - y[valid]).square().mean()))


def sign_accuracy(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return fraction of observations with matching O-information sign."""
    valid = torch.isfinite(x) & torch.isfinite(y) & (y != 0)
    return float((torch.sign(x[valid]) == torch.sign(y[valid])).to(torch.float64).mean())


def finite_mean(values: list[float]) -> float:
    """Return mean after dropping non-finite values."""
    tensor = torch.tensor(values, dtype=torch.float64)
    tensor = tensor[torch.isfinite(tensor)]
    return float(tensor.mean()) if tensor.numel() else float("nan")


def sample_model(sample_count: int, generator: torch.Generator) -> torch.Tensor:
    """Draw samples from the exact validation model."""
    probabilities = exact_joint_probabilities()
    states = torch.multinomial(
        probabilities,
        sample_count,
        replacement=True,
        generator=generator,
    )
    return ((states[:, None] >> torch.arange(N_VARIABLES)) & 1).to(torch.uint8)


def main() -> None:
    """Compare current and coherent direct local O estimators across sample sizes."""
    exact_joint = exact_joint_probabilities()
    support = support_bits()
    exact_support_local_o = direct_local_o(exact_joint, support)
    exact_global_o = float((exact_joint * exact_support_local_o).sum())

    print("[MODEL]")
    signals = 1.0 - 2.0 * NOISE
    pairwise = torch.tensor(
        [float(signals[i] * signals[j]) for i, j in itertools.combinations(range(N_VARIABLES), 2)],
        dtype=torch.float64,
    )
    print(f"pairwise_corr_min={float(pairwise.min()):.8f}")
    print(f"pairwise_corr_median={float(pairwise.median()):.8f}")
    print(f"pairwise_corr_max={float(pairwise.max()):.8f}")
    print(f"ground_truth_O={exact_global_o:.12f}")
    print(f"ground_truth_local_O_sd={float(torch.sqrt((exact_joint * (exact_support_local_o - exact_global_o).square()).sum())):.12f}")

    generator = torch.Generator().manual_seed(SEED)
    max_n = max(SAMPLE_SIZES)

    corr_current: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    corr_direct_empirical: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    corr_joint_shrinkage: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    rmse_current: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    rmse_joint_shrinkage: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    sign_current: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    sign_joint_shrinkage: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    corr_current_vs_shrinkage: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    global_error_current_empirical: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    global_error_current_shrinkage: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    global_error_joint_shrinkage: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}
    lambdas: dict[int, list[float]] = {n: [] for n in SAMPLE_SIZES}

    max_empirical_direct_residual = 0.0
    max_shrinkage_entropy_residual = 0.0

    for replicate in range(N_REPLICATES):
        full_data = sample_model(max_n, generator)
        for n in SAMPLE_SIZES:
            data = full_data[:n]
            truth = exact_local_o(data)

            current = current_empirical_local_o(data)
            empirical = empirical_full_joint(data)
            direct_empirical = direct_local_o(empirical, data)
            residual = float(torch.max(torch.abs(current - direct_empirical)))
            max_empirical_direct_residual = max(max_empirical_direct_residual, residual)

            shrinkage_joint, shrinkage_lambda = shrink_full_joint(data)
            direct_shrinkage = direct_local_o(shrinkage_joint, data)
            lambdas[n].append(shrinkage_lambda)

            repo_h_shrinkage = float(
                dthoi.estimate_entropy(
                    data,
                    FULL_SET.unsqueeze(0),
                    entropy_estimator="shrinkage",
                )[0, 0]
            )
            max_shrinkage_entropy_residual = max(
                max_shrinkage_entropy_residual,
                abs(repo_h_shrinkage - entropy(shrinkage_joint)),
            )

            corr_current[n].append(pearson(current, truth))
            corr_direct_empirical[n].append(pearson(direct_empirical, truth))
            corr_joint_shrinkage[n].append(pearson(direct_shrinkage, truth))
            rmse_current[n].append(rmse(current, truth))
            rmse_joint_shrinkage[n].append(rmse(direct_shrinkage, truth))
            sign_current[n].append(sign_accuracy(current, truth))
            sign_joint_shrinkage[n].append(sign_accuracy(direct_shrinkage, truth))
            corr_current_vs_shrinkage[n].append(pearson(current, direct_shrinkage))

            global_error_current_empirical[n].append(
                current_global_o(data, "empirical") - exact_global_o
            )
            global_error_current_shrinkage[n].append(
                current_global_o(data, "shrinkage") - exact_global_o
            )
            coherent_global_o = float(
                (shrinkage_joint * direct_local_o(shrinkage_joint, support)).sum()
            )
            global_error_joint_shrinkage[n].append(coherent_global_o - exact_global_o)

        print(f"completed_replicate={replicate + 1}/{N_REPLICATES}")

    print("\n[CONTRACTS]")
    print(f"empirical_direct_max_abs_diff={max_empirical_direct_residual:.3e}")
    print(f"shrinkage_joint_entropy_vs_repo_max_abs_diff={max_shrinkage_entropy_residual:.3e}")

    print("\n[LOCAL_O_ACCURACY]")
    print("N,current_corr,direct_empirical_corr,joint_shrinkage_corr,current_rmse,joint_shrinkage_rmse,current_sign_acc,joint_shrinkage_sign_acc,current_vs_joint_shrinkage_corr,mean_lambda")
    for n in SAMPLE_SIZES:
        print(
            f"{n},"
            f"{finite_mean(corr_current[n]):.8f},"
            f"{finite_mean(corr_direct_empirical[n]):.8f},"
            f"{finite_mean(corr_joint_shrinkage[n]):.8f},"
            f"{finite_mean(rmse_current[n]):.8f},"
            f"{finite_mean(rmse_joint_shrinkage[n]):.8f},"
            f"{finite_mean(sign_current[n]):.8f},"
            f"{finite_mean(sign_joint_shrinkage[n]):.8f},"
            f"{finite_mean(corr_current_vs_shrinkage[n]):.8f},"
            f"{finite_mean(lambdas[n]):.8f}"
        )

    print("\n[GLOBAL_O_RMSE]")
    print("N,current_empirical,current_separate_shrinkage,coherent_joint_shrinkage")
    for n in SAMPLE_SIZES:
        values = []
        for source in (
            global_error_current_empirical[n],
            global_error_current_shrinkage[n],
            global_error_joint_shrinkage[n],
        ):
            values.append(math.sqrt(finite_mean([error * error for error in source])))
        print(f"{n},{values[0]:.8f},{values[1]:.8f},{values[2]:.8f}")


if __name__ == "__main__":
    main()
