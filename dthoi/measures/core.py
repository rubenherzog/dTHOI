from __future__ import annotations

from typing import TypeAlias

import torch

from ..data import DiscreteData, DiscreteInput
from ..entropy.base import CountingEntropyProvider
from ..entropy.cache import EntropyCache
from ..entropy.counting import CountMode
from ..entropy.estimators import CountEntropyEstimator
from ..subsets import (
    _masks_from_canonical_subsets,
    canonicalize_subsets,
    leave_one_out_subsets,
)

LocalMeasureValues: TypeAlias = tuple[torch.Tensor, ...]
MeasureOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, LocalMeasureValues]

DEFAULT_MAX_LOCAL_VALUES_PER_BATCH = 1_000_000


def local_sets_per_batch(
    data: DiscreteData,
    max_local_values_per_batch: int,
) -> int:
    """Resolve a safe number of variable sets for sample-resolved calculations.

    Parameters
    ----------
    data
        Prepared datasets whose sample counts determine local output size.
    max_local_values_per_batch
        Maximum target for ``variable sets × total samples`` in one local
        calculation batch. The four returned measures and temporary entropy
        terms introduce additional constant factors, but this product is the
        dominant data-dependent scaling term.

    Returns
    -------
    int
        Positive number of variable sets to evaluate together. A single set is
        always allowed even when its sample count alone exceeds the target.
    """
    if max_local_values_per_batch <= 0:
        raise ValueError("max_local_values_per_batch must be positive.")
    total_samples = sum(data.sample_counts)
    return max(1, max_local_values_per_batch // total_samples)


def _finish_measure_channels(values: torch.Tensor) -> None:
    """Fill O-information and S-information from TC and DTC in place."""
    values[..., 2] = values[..., 0] - values[..., 1]
    values[..., 3] = values[..., 0] + values[..., 1]


def _combine_global_terms(
    h_joint: torch.Tensor,
    h_single: torch.Tensor,
    h_loo: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """Combine shared entropy terms into the four global information measures."""
    values = torch.empty((*h_joint.shape, 4), dtype=torch.float64, device=h_joint.device)
    values[..., 0] = h_single - h_joint
    values[..., 1] = h_loo - (order - 1) * h_joint
    _finish_measure_channels(values)
    return values


def _accumulate_local_entropy_terms(
    provider: CountingEntropyProvider,
    term_subsets: torch.Tensor,
    parent_indices: torch.Tensor,
    global_channel: torch.Tensor,
    local_channels: tuple[torch.Tensor, ...],
    *,
    max_local_values_per_batch: int,
) -> None:
    """Accumulate unique entropy terms without materializing ``[B, K, T]``.

    Repeated singleton or leave-one-out variable sets are deduplicated across
    the complete parent batch. Each unique entropy term is estimated once,
    then scattered to every parent variable set that uses it. Both estimation
    and scatter expansion are chunked according to the configured
    ``variable sets × samples`` target.
    """
    terms_per_batch = local_sets_per_batch(
        provider.data,
        max_local_values_per_batch,
    )
    masks = _masks_from_canonical_subsets(term_subsets)
    parent_list = parent_indices.detach().cpu().tolist()

    unique_subsets: list[torch.Tensor] = []
    unique_index: dict[int, int] = {}
    parents_by_unique: list[list[int]] = []

    for mask, subset, parent in zip(masks, term_subsets, parent_list, strict=True):
        index = unique_index.get(mask)
        if index is None:
            index = len(unique_subsets)
            unique_index[mask] = index
            unique_subsets.append(subset)
            parents_by_unique.append([])
        parents_by_unique[index].append(parent)

    for unique_start in range(0, len(unique_subsets), terms_per_batch):
        unique_stop = min(unique_start + terms_per_batch, len(unique_subsets))
        entropy_subsets = torch.stack(unique_subsets[unique_start:unique_stop])
        term_entropy, term_local = provider.entropy_with_local(entropy_subsets)

        source_rows: list[int] = []
        target_parents: list[int] = []
        for source_row, parents in enumerate(
            parents_by_unique[unique_start:unique_stop]
        ):
            source_rows.extend([source_row] * len(parents))
            target_parents.extend(parents)

        for occurrence_start in range(0, len(source_rows), terms_per_batch):
            occurrence_stop = min(
                occurrence_start + terms_per_batch,
                len(source_rows),
            )
            sources = torch.tensor(
                source_rows[occurrence_start:occurrence_stop],
                dtype=torch.long,
                device=provider.data.device,
            )
            parents = torch.tensor(
                target_parents[occurrence_start:occurrence_stop],
                dtype=torch.long,
                device=provider.data.device,
            )
            global_channel.index_add_(
                0,
                parents,
                term_entropy.index_select(0, sources),
            )
            for dataset_index, local_values in enumerate(term_local):
                local_channels[dataset_index].index_add_(
                    0,
                    parents,
                    local_values.index_select(0, sources),
                )


def _measures_with_local_from_provider(
    provider: CountingEntropyProvider,
    subsets: torch.Tensor,
    *,
    max_local_values_per_batch: int,
) -> tuple[torch.Tensor, LocalMeasureValues]:
    """Compute global and sample-resolved measures for one bounded set batch."""
    batch_size, order = subsets.shape
    n_datasets = provider.data.n_datasets
    device = provider.data.device

    h_joint, h_joint_local = provider.entropy_with_local(subsets)

    values = torch.empty(
        (batch_size, n_datasets, 4),
        dtype=torch.float64,
        device=device,
    )
    values[..., 0] = -h_joint
    values[..., 1] = -(order - 1) * h_joint

    local_values = tuple(
        torch.empty(
            (batch_size, dataset.shape[0], 4),
            dtype=torch.float64,
            device=device,
        )
        for dataset in provider.data.datasets
    )
    for dataset_index, joint_local in enumerate(h_joint_local):
        local_values[dataset_index][..., 0] = -joint_local
        local_values[dataset_index][..., 1] = -(order - 1) * joint_local

    parent_indices = torch.arange(batch_size, device=device).repeat_interleave(order)

    singleton_subsets = subsets.reshape(batch_size * order, 1)
    _accumulate_local_entropy_terms(
        provider,
        singleton_subsets,
        parent_indices,
        values[..., 0],
        tuple(dataset_values[..., 0] for dataset_values in local_values),
        max_local_values_per_batch=max_local_values_per_batch,
    )

    loo_subsets = leave_one_out_subsets(subsets)
    _accumulate_local_entropy_terms(
        provider,
        loo_subsets,
        parent_indices,
        values[..., 1],
        tuple(dataset_values[..., 1] for dataset_values in local_values),
        max_local_values_per_batch=max_local_values_per_batch,
    )

    _finish_measure_channels(values)
    for dataset_values in local_values:
        _finish_measure_channels(dataset_values)

    return values, local_values


def measures_from_provider(
    provider: CountingEntropyProvider,
    subsets: torch.Tensor,
    *,
    local_values: bool = False,
    max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
) -> MeasureOutput:
    """Compute TC, DTC, O-information, and S-information from subset entropies.

    Parameters
    ----------
    provider
        Prepared entropy provider used for all required marginal entropies.
    subsets
        Fixed-order variable subsets with shape ``[B, K]`` and ``K >= 2``.
    local_values
        If ``True``, also return one value per observation for every measure.
        The default ``False`` preserves the global-only fast path.
    max_local_values_per_batch
        Target upper bound for ``variable sets × total samples`` in temporary
        local entropy batches. This controls the multiplication by sample count
        without changing the requested scientific variable sets.

    Returns
    -------
    torch.Tensor or tuple
        Global measures have shape ``[B, D, 4]`` in the fixed order
        ``(TC, DTC, O-information, S-information)``. With ``local_values=True``,
        a second tuple contains one ``[B, T_d, 4]`` tensor per dataset.

    Notes
    -----
    Local and global measures use the same entropy identities. TC and DTC are
    accumulated from joint, singleton, and leave-one-out entropy terms; O- and
    S-information are then derived as ``TC - DTC`` and ``TC + DTC``. No
    measure-specific entropy estimation pipeline is introduced.
    """
    subsets = canonicalize_subsets(subsets, provider.data.n_variables)
    batch_size, order = subsets.shape
    if order < 2:
        raise ValueError("Multivariate measures require subset order >= 2.")

    subsets = subsets.to(provider.data.device)

    if local_values:
        return _measures_with_local_from_provider(
            provider,
            subsets,
            max_local_values_per_batch=max_local_values_per_batch,
        )

    h_joint = provider.entropy(subsets)

    singleton_subsets = subsets.reshape(batch_size * order, 1)
    h_single = provider.entropy(singleton_subsets).reshape(batch_size, order, -1).sum(dim=1)

    loo_subsets = leave_one_out_subsets(subsets)
    h_loo = provider.entropy(loo_subsets).reshape(batch_size, order, -1).sum(dim=1)

    return _combine_global_terms(h_joint, h_single, h_loo, order)


def nplets_measures(
    X: DiscreteInput,
    subsets: torch.Tensor,
    *,
    estimator: str | CountEntropyEstimator = "plugin",
    count_mode: CountMode = "auto",
    cache: EntropyCache | None = None,
    device: torch.device | str | None = None,
    local_values: bool = False,
    max_local_values_per_batch: int = DEFAULT_MAX_LOCAL_VALUES_PER_BATCH,
) -> MeasureOutput:
    """Compute higher-order information measures for explicit variable sets.

    This is an internal tensor-level function. Scientific users should normally
    call :func:`dthoi.information_measures`.

    Parameters
    ----------
    X
        Discrete observations accepted by :func:`dthoi.prepare_data`.
    subsets
        Variable sets with shape ``[B, K]`` or one set ``[K]``.
    estimator
        Entropy estimator name or estimator instance.
    count_mode
        ``"dense"``, ``"sparse"``, or ``"auto"`` counting strategy.
    cache
        Optional entropy cache to reuse across calls.
    device
        Destination device for the observations.
    local_values
        Whether to also calculate sample-resolved TC, DTC, O-information, and
        S-information.
    max_local_values_per_batch
        Target upper bound for ``variable sets × total samples`` during local
        calculations. Explicit variable sets are subdivided automatically when
        needed, then restored to their original order in the returned tensors.

    Returns
    -------
    torch.Tensor or tuple
        Global measures with shape ``[B, D, 4]``. If local values are requested,
        also returns one ``[B, T_d, 4]`` tensor per dataset.
    """
    provider = CountingEntropyProvider(
        X,
        estimator=estimator,
        count_mode=count_mode,
        cache=cache,
        device=device,
    )
    subsets = canonicalize_subsets(subsets, provider.data.n_variables).to(provider.data.device)

    if not local_values:
        return measures_from_provider(provider, subsets)

    sets_per_batch = local_sets_per_batch(provider.data, max_local_values_per_batch)
    values_out = torch.empty(
        (subsets.shape[0], provider.data.n_datasets, 4),
        dtype=torch.float64,
        device=provider.data.device,
    )
    local_out = tuple(
        torch.empty(
            (subsets.shape[0], dataset.shape[0], 4),
            dtype=torch.float64,
            device=provider.data.device,
        )
        for dataset in provider.data.datasets
    )

    for start in range(0, subsets.shape[0], sets_per_batch):
        stop = min(start + sets_per_batch, subsets.shape[0])
        output = measures_from_provider(
            provider,
            subsets[start:stop],
            local_values=True,
            max_local_values_per_batch=max_local_values_per_batch,
        )
        assert isinstance(output, tuple)
        values, local = output
        values_out[start:stop] = values
        for dataset_index, dataset_values in enumerate(local):
            local_out[dataset_index][start:stop] = dataset_values

    return values_out, local_out
