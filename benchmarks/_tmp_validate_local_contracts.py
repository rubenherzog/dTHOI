from __future__ import annotations

import sys
from pathlib import Path

import torch

import dthoi
from dthoi.entropy.base import CountingEntropyProvider

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tmp_validate_weak_correlations as validation


def main() -> None:
    """Verify exact local-to-global reconstruction on the validation model."""
    probabilities = validation.subset_probabilities(validation.FULL_SET)
    states = torch.arange(1 << validation.N_VARIABLES, dtype=torch.long)
    support_data = (
        (states[:, None] >> torch.arange(validation.N_VARIABLES)) & 1
    ).to(torch.uint8)
    exact_global = validation.exact_global_metrics()
    exact_local = validation.exact_local_metrics(support_data)

    max_truth_residual = 0.0
    for metric in validation.METRICS:
        recovered = float((probabilities * exact_local[metric]).sum())
        residual = abs(recovered - exact_global[metric])
        max_truth_residual = max(max_truth_residual, residual)
    assert max_truth_residual < 1e-12

    generator = torch.Generator().manual_seed(validation.SEED + 1)
    data = validation.sample_model(4096, generator)
    max_repo_residual = 0.0
    max_identity_residual = 0.0

    for estimator in validation.LOCAL_ESTIMATORS:
        provider = CountingEntropyProvider(data, estimator=estimator, count_mode="auto")
        entropy, local_entropy = provider.entropy_with_local(
            validation.FULL_SET.unsqueeze(0)
        )
        entropy_residual = abs(float(local_entropy[0][0].mean() - entropy[0, 0]))
        max_repo_residual = max(max_repo_residual, entropy_residual)

        result = dthoi.information_measures(
            data,
            validation.FULL_SET.unsqueeze(0),
            entropy_estimator=estimator,
            local_values=True,
        )
        local = result.local.values[0][0]
        for index in range(4):
            residual = abs(float(local[:, index].mean() - result.values[0, 0, index]))
            max_repo_residual = max(max_repo_residual, residual)

        o_identity = torch.max(torch.abs(local[:, 2] - (local[:, 0] - local[:, 1])))
        s_identity = torch.max(torch.abs(local[:, 3] - (local[:, 0] + local[:, 1])))
        max_identity_residual = max(
            max_identity_residual,
            float(o_identity),
            float(s_identity),
        )

    assert max_repo_residual < 1e-12
    assert max_identity_residual < 1e-12

    for estimator in ("shrinkage", "pitman_yor", "ansb"):
        provider = CountingEntropyProvider(data, estimator=estimator, count_mode="auto")
        try:
            provider.entropy_with_local(validation.FULL_SET.unsqueeze(0))
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{estimator} unexpectedly exposed local values")

    print("[LOCAL_CONTRACT_CHECK]")
    print(f"ground_truth_weighted_mean_max_residual={max_truth_residual:.3e}")
    print(f"repo_local_mean_max_residual={max_repo_residual:.3e}")
    print(f"local_O_S_identity_max_residual={max_identity_residual:.3e}")
    print("unsupported_local_estimators=shrinkage,pitman_yor,ansb")


if __name__ == "__main__":
    main()
