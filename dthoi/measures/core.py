from __future__ import annotations

import torch

from ..data import DiscreteInput
from ..entropy.base import CountingEntropyProvider
from ..entropy.cache import EntropyCache
from ..entropy.counting import CountMode
from ..entropy.estimators import CountEntropyEstimator
from ..subsets import canonicalize_subsets, leave_one_out_subsets


def measures_from_provider(provider: CountingEntropyProvider, subsets: torch.Tensor) -> torch.Tensor:
    """Compute TC, DTC, O-information, and S-information from subset entropies.

    Parameters
    ----------
    provider
        Prepared entropy provider used for all required marginal entropies.
    subsets
        Fixed-order variable subsets with shape ``[B, K]`` and ``K >= 2``.

    Returns
    -------
    torch.Tensor
        Measures with shape ``[B, D, 4]`` in the fixed order
        ``(TC, DTC, O-information, S-information)``.
    """
    subsets = canonicalize_subsets(subsets, provider.data.n_variables)
    batch_size, order = subsets.shape
    if order < 2:
        raise ValueError("Multivariate measures require subset order >= 2.")

    subsets = subsets.to(provider.data.device)
    h_joint = provider.entropy(subsets)

    singleton_subsets = subsets.reshape(batch_size * order, 1)
    h_single = provider.entropy(singleton_subsets).reshape(batch_size, order, -1).sum(dim=1)

    loo_subsets = leave_one_out_subsets(subsets)
    h_loo = provider.entropy(loo_subsets).reshape(batch_size, order, -1).sum(dim=1)

    tc = h_single - h_joint
    dtc = h_loo - (order - 1) * h_joint
    o_information = tc - dtc
    s_information = tc + dtc

    return torch.stack((tc, dtc, o_information, s_information), dim=-1)


def nplets_measures(
    X: DiscreteInput,
    subsets: torch.Tensor,
    *,
    estimator: str | CountEntropyEstimator = "plugin",
    count_mode: CountMode = "auto",
    cache: EntropyCache | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute higher-order information measures for an explicit subset batch.

    Parameters
    ----------
    X
        Discrete observations accepted by :func:`dthoi.prepare_discrete_data`.
    subsets
        Variable subsets with shape ``[B, K]`` or one subset ``[K]``.
    estimator
        Entropy estimator name or estimator instance.
    count_mode
        ``"dense"``, ``"sparse"``, or ``"auto"`` counting strategy.
    cache
        Optional entropy cache to reuse across calls.
    device
        Destination device for the observations.

    Returns
    -------
    torch.Tensor
        Measures with shape ``[B, D, 4]`` ordered as TC, DTC, O-information,
        and S-information.
    """
    provider = CountingEntropyProvider(
        X,
        estimator=estimator,
        count_mode=count_mode,
        cache=cache,
        device=device,
    )
    return measures_from_provider(provider, subsets)
