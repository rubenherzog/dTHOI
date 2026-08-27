import math

import torch

from dthoi.entropy.base import CountingEntropyProvider
from dthoi.entropy.cache import EntropyCache
from dthoi.entropy.estimators import MillerMadowEstimator, PluginEstimator


def full_binary_table(k: int) -> torch.Tensor:
    rows = []
    for state in range(1 << k):
        rows.append([(state >> bit) & 1 for bit in range(k)])
    return torch.tensor(rows, dtype=torch.uint8)


def test_plugin_entropy_exact_for_uniform_binary_subsets():
    X = full_binary_table(4)
    provider = CountingEntropyProvider(X, estimator="plugin", count_mode="dense")
    cases = [([0], 1.0), ([0, 1], 2.0), ([0, 1, 2], 3.0), ([0, 1, 2, 3], 4.0)]
    for subset, expected in cases:
        actual = provider.entropy(torch.tensor([subset]))[:, 0]
        assert torch.allclose(actual, torch.tensor([expected], dtype=torch.float64))


def test_dense_and_sparse_counting_match():
    X = torch.tensor([
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ], dtype=torch.uint8)
    subsets = torch.tensor([[0, 1, 2], [1, 2, 3]])

    dense = CountingEntropyProvider(X, count_mode="dense").entropy(subsets)
    sparse = CountingEntropyProvider(X, count_mode="sparse").entropy(subsets)
    assert torch.allclose(dense, sparse, atol=1e-12)


def test_entropy_cache_reuses_canonical_subset():
    X = full_binary_table(3)
    cache = EntropyCache()
    provider = CountingEntropyProvider(X, cache=cache)

    first = provider.entropy(torch.tensor([[2, 0, 1]]))
    stats_after_first = (cache.stats.hits, cache.stats.misses)
    second = provider.entropy(torch.tensor([[1, 2, 0]]))

    assert torch.equal(first, second)
    assert stats_after_first == (0, 1)
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_miller_madow_is_plugin_plus_expected_correction():
    counts = torch.tensor([[3, 1, 0, 0]], dtype=torch.int64)
    plugin = PluginEstimator().entropy_from_dense(counts)
    mm = MillerMadowEstimator().entropy_from_dense(counts)
    expected = plugin + torch.tensor([(2 - 1) / (2 * 4 * math.log(2))], dtype=torch.float64)
    assert torch.allclose(mm, expected, atol=1e-12)
