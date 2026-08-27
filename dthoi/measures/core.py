from __future__ import annotations

import torch

from ..data import PreparedDiscreteData
from ..entropy.base import CountingEntropyProvider
from ..entropy.cache import EntropyCache
from ..entropy.counting import CountMode
from ..entropy.estimators import CountEntropyEstimator
from ..subsets import canonicalize_subsets

MEASURE_NAMES = ("tc", "dtc", "o", "s")


def measures_from_provider(provider: CountingEntropyProvider, subsets: torch.Tensor) -> torch.Tensor:
    """Compute TC, DTC, O-information and S-information from cached entropies."""

    subsets = canonicalize_subsets(subsets, provider.data.n_variables)
    B, K = subsets.shape
    if K < 2:
        raise ValueError("Multivariate measures require subset order >= 2.")

    device = provider.data.device
    subsets = subsets.to(device)

    h_joint = provider.entropy(subsets)  # [B, D]

    singleton_subsets = subsets.reshape(B * K, 1)
    h_single = provider.entropy(singleton_subsets).reshape(B, K, -1).sum(dim=1)

    leave_one_out = torch.stack(
        [torch.cat((subsets[:, :i], subsets[:, i + 1 :]), dim=1) for i in range(K)],
        dim=1,
    ).reshape(B * K, K - 1)
    h_loo = provider.entropy(leave_one_out).reshape(B, K, -1).sum(dim=1)

    tc = h_single - h_joint
    dtc = h_loo - (K - 1) * h_joint
    o = tc - dtc
    s = tc + dtc

    return torch.stack((tc, dtc, o, s), dim=-1)


def nplets_measures(
    X: PreparedDiscreteData | torch.Tensor | list[torch.Tensor],
    subsets: torch.Tensor,
    *,
    estimator: str | CountEntropyEstimator = "plugin",
    count_mode: CountMode = "auto",
    cache: EntropyCache | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    provider = CountingEntropyProvider(
        X,
        estimator=estimator,
        count_mode=count_mode,
        cache=cache,
        device=device,
    )
    return measures_from_provider(provider, subsets)
