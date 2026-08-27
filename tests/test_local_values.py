import torch

import dthoi
from dthoi.entropy.base import CountingEntropyProvider
from dthoi.measures.core import measures_from_provider, nplets_measures
from dthoi.subsets import canonicalize_subsets


def _data() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=torch.uint8,
    )


def test_empirical_local_entropy_mean_matches_global_dense_and_sparse():
    subsets = torch.tensor([[0, 1, 2], [1, 2, 3]])
    for mode in ("dense", "sparse"):
        provider = CountingEntropyProvider(
            _data(), estimator="empirical", count_mode=mode
        )
        entropy, local = provider.entropy_with_local(subsets)
        torch.testing.assert_close(
            local[0].mean(dim=1), entropy[:, 0], rtol=0.0, atol=1e-12
        )


def test_miller_madow_local_entropy_mean_matches_corrected_global():
    provider = CountingEntropyProvider(
        _data(), estimator="miller_madow", count_mode="sparse"
    )
    entropy, local = provider.entropy_with_local(torch.tensor([[0, 1, 2]]))
    torch.testing.assert_close(
        local[0].mean(dim=1), entropy[:, 0], rtol=0.0, atol=1e-12
    )


def test_all_local_measures_average_to_global_values():
    output = nplets_measures(
        _data(),
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        count_mode="sparse",
        local_values=True,
        max_local_values_per_batch=7,
    )
    assert isinstance(output, tuple)
    values, local = output
    torch.testing.assert_close(
        local[0].mean(dim=1), values[:, 0, :], rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        local[0][..., 2],
        local[0][..., 0] - local[0][..., 1],
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        local[0][..., 3],
        local[0][..., 0] + local[0][..., 1],
        rtol=0.0,
        atol=1e-12,
    )


def test_miller_madow_local_measures_average_to_corrected_global_values():
    output = nplets_measures(
        _data(),
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        estimator="miller_madow",
        count_mode="sparse",
        local_values=True,
        max_local_values_per_batch=7,
    )
    assert isinstance(output, tuple)
    values, local = output
    torch.testing.assert_close(
        local[0].mean(dim=1), values[:, 0, :], rtol=0.0, atol=1e-12
    )


def test_dense_and_sparse_local_measures_match():
    subsets = torch.tensor([[0, 1, 2], [1, 2, 3]])
    dense = nplets_measures(
        _data(),
        subsets,
        count_mode="dense",
        local_values=True,
        max_local_values_per_batch=7,
    )
    sparse = nplets_measures(
        _data(),
        subsets,
        count_mode="sparse",
        local_values=True,
        max_local_values_per_batch=7,
    )
    assert isinstance(dense, tuple) and isinstance(sparse, tuple)
    torch.testing.assert_close(dense[0], sparse[0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(dense[1][0], sparse[1][0], rtol=0.0, atol=1e-12)


def test_repeated_variable_sets_preserve_local_rows():
    output = nplets_measures(
        _data(),
        torch.tensor(
            [
                [0, 1, 2],
                [2, 1, 0],
                [0, 1, 2],
            ]
        ),
        count_mode="sparse",
        local_values=True,
        max_local_values_per_batch=7,
    )
    assert isinstance(output, tuple)
    values, local = output
    torch.testing.assert_close(values[0], values[1], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(values[0], values[2], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(local[0][0], local[0][1], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(local[0][0], local[0][2], rtol=0.0, atol=1e-12)


def test_unequal_dataset_lengths_remain_separate():
    first = _data()
    second = _data()[:4]
    result = dthoi.information_measures(
        [first, second],
        [[0, 1, 2]],
        local_values=True,
        max_local_values_per_batch=4,
    )
    assert result.local is not None
    assert result.local.values[0].shape == (1, 7, 4)
    assert result.local.values[1].shape == (1, 4, 4)
    torch.testing.assert_close(
        result.local.values[0].mean(dim=1),
        result.values[:, 0, :],
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.local.values[1].mean(dim=1),
        result.values[:, 1, :],
        rtol=0.0,
        atol=1e-12,
    )


def test_public_api_exposes_named_local_views_only_when_requested():
    no_local = dthoi.information_measures(_data(), [[0, 1, 2]])
    assert no_local.local is None

    result = dthoi.information_measures(_data(), [[0, 1, 2]], local_values=True)
    assert isinstance(result.local, dthoi.LocalInformationMeasures)
    assert len(result.local.total_correlation) == 1
    assert result.local.total_correlation[0].shape == (1, 7)
    torch.testing.assert_close(
        result.local.o_information[0], result.local.values[0][..., 2]
    )


def test_analyze_orders_reduces_effective_batch_when_samples_make_locals_large():
    results = dthoi.analyze_orders(
        _data(),
        min_order=3,
        max_order=3,
        sets_per_batch=10,
        local_values=True,
        max_local_values_per_batch=14,
    )
    assert results
    assert all(item.variable_sets.shape[0] <= 2 for item in results)
    assert all(item.information.local is not None for item in results)
    assert sum(item.variable_sets.shape[0] for item in results) == 4


class _RecordingProvider(CountingEntropyProvider):
    """Record local entropy requests while preserving provider behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_requests: list[list[tuple[int, ...]]] = []

    def entropy_with_local(self, subsets):
        canonical = canonicalize_subsets(subsets, self.data.n_variables)
        self.local_requests.append(
            [tuple(row.tolist()) for row in canonical.detach().cpu()]
        )
        return super().entropy_with_local(subsets)


def test_shared_local_entropy_terms_are_deduplicated_across_chunks():
    provider = _RecordingProvider(_data(), count_mode="sparse")
    output = measures_from_provider(
        provider,
        torch.tensor([[0, 1, 2], [0, 1, 3]]),
        local_values=True,
        max_local_values_per_batch=14,
    )
    assert isinstance(output, tuple)

    singleton_batches = [
        batch for batch in provider.local_requests if batch and len(batch[0]) == 1
    ]
    loo_batches = [
        batch for batch in provider.local_requests if batch and len(batch[0]) == 2
    ]

    assert len(singleton_batches) > 1
    assert len(loo_batches) > 1

    singletons = [subset for batch in singleton_batches for subset in batch]
    loo = [subset for batch in loo_batches for subset in batch]

    assert len(singletons) == len(set(singletons)) == 4
    assert len(loo) == len(set(loo)) == 5
