import torch

import dthoi
from dthoi.entropy.base import CountingEntropyProvider
from dthoi.measures.core import nplets_measures


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
