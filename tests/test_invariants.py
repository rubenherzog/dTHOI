import torch

from dthoi.batch import multi_order_measures
from dthoi.entropy.base import CountingEntropyProvider
from dthoi.entropy.cache import EntropyCache
from dthoi.measures.core import nplets_measures


class _NullCache:
    def get(self, key):
        return None

    def put(self, key, value):
        pass


def _random_binary(T=257, N=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (T, N), dtype=torch.uint8, generator=generator)


def _flatten_batches(batches):
    out = {}
    for subsets, values in batches:
        for subset, value in zip(subsets, values, strict=True):
            out[tuple(subset.tolist())] = value
    return out


def test_subset_variable_order_invariance():
    X = _random_binary()
    a = nplets_measures(X, torch.tensor([[0, 2, 4, 6]]), count_mode="dense")
    b = nplets_measures(X, torch.tensor([[6, 0, 4, 2]]), count_mode="dense")
    torch.testing.assert_close(a, b, rtol=0.0, atol=1e-12)


def test_sample_order_invariance():
    X = _random_binary(seed=1)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(10))
    subsets = torch.tensor([[0, 1, 4, 7], [2, 3, 5, 6]])
    a = nplets_measures(X, subsets, count_mode="sparse")
    b = nplets_measures(X[perm], subsets, count_mode="sparse")
    torch.testing.assert_close(a, b, rtol=0.0, atol=1e-12)


def test_cache_and_no_cache_match():
    X = _random_binary(seed=2)
    subsets = torch.tensor([[0, 1, 2, 3], [1, 3, 5, 7], [0, 2, 4, 6]])
    cached = CountingEntropyProvider(X, cache=EntropyCache(), count_mode="dense").entropy(subsets)
    uncached = CountingEntropyProvider(X, cache=_NullCache(), count_mode="dense").entropy(subsets)
    torch.testing.assert_close(cached, uncached, rtol=0.0, atol=1e-12)


def test_repeated_subsets_return_identical_values():
    X = _random_binary(seed=3)
    subsets = torch.tensor([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [0, 1, 2, 3],
        [1, 4, 5, 7],
    ])
    values = nplets_measures(X, subsets, count_mode="dense")
    torch.testing.assert_close(values[0], values[1], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(values[0], values[2], rtol=0.0, atol=1e-12)


def test_dense_sparse_and_auto_match():
    X = _random_binary(T=311, N=9, seed=4)
    subsets = torch.tensor([[0, 1, 2, 3, 4], [1, 3, 5, 6, 8], [0, 2, 4, 7, 8]])
    dense = nplets_measures(X, subsets, count_mode="dense")
    sparse = nplets_measures(X, subsets, count_mode="sparse")
    auto = nplets_measures(X, subsets, count_mode="auto")
    torch.testing.assert_close(dense, sparse, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(dense, auto, rtol=0.0, atol=1e-12)


def test_batch_size_invariance():
    X = _random_binary(T=129, N=6, seed=5)
    small = _flatten_batches(
        multi_order_measures(X, min_order=2, max_order=4, batch_size=2, count_mode="dense")
    )
    large = _flatten_batches(
        multi_order_measures(X, min_order=2, max_order=4, batch_size=64, count_mode="dense")
    )
    assert small.keys() == large.keys()
    for key in small:
        torch.testing.assert_close(small[key], large[key], rtol=0.0, atol=1e-12)


def test_multiple_ragged_datasets_are_independently_sample_order_invariant():
    x1 = _random_binary(T=101, N=6, seed=6)
    x2 = _random_binary(T=173, N=6, seed=7)
    p1 = torch.randperm(x1.shape[0], generator=torch.Generator().manual_seed(8))
    p2 = torch.randperm(x2.shape[0], generator=torch.Generator().manual_seed(9))
    subsets = torch.tensor([[0, 1, 2, 3], [1, 3, 4, 5]])
    a = nplets_measures([x1, x2], subsets, count_mode="sparse")
    b = nplets_measures([x1[p1], x2[p2]], subsets, count_mode="sparse")
    torch.testing.assert_close(a, b, rtol=0.0, atol=1e-12)


def test_constant_variables_have_zero_measures():
    X = torch.zeros((37, 5), dtype=torch.uint8)
    values = nplets_measures(X, torch.tensor([[0, 1, 2, 3, 4]]), count_mode="sparse")
    torch.testing.assert_close(values, torch.zeros_like(values), rtol=0.0, atol=1e-12)


def test_duplicate_variables_match_redundant_analytic_case():
    base = torch.tensor([0, 1, 0, 1, 1, 1], dtype=torch.uint8)
    X = base[:, None].repeat(1, 4)
    values = nplets_measures(X, torch.tensor([[0, 1, 2, 3]]), count_mode="sparse")[0, 0]
    h = -(2 / 6) * torch.log2(torch.tensor(2 / 6, dtype=torch.float64))
    h -= (4 / 6) * torch.log2(torch.tensor(4 / 6, dtype=torch.float64))
    expected = torch.tensor([3 * h, h, 2 * h, 4 * h], dtype=torch.float64)
    torch.testing.assert_close(values, expected, rtol=0.0, atol=1e-12)


def test_different_orders_can_share_one_provider_cache():
    X = _random_binary(T=211, N=7, seed=10)
    provider = CountingEntropyProvider(X, cache=EntropyCache(), count_mode="auto")
    h2 = provider.entropy(torch.tensor([[0, 1], [2, 3]]))
    h4 = provider.entropy(torch.tensor([[0, 1, 2, 3], [1, 3, 4, 6]]))
    fresh2 = CountingEntropyProvider(X, count_mode="auto").entropy(torch.tensor([[0, 1], [2, 3]]))
    fresh4 = CountingEntropyProvider(X, count_mode="auto").entropy(
        torch.tensor([[0, 1, 2, 3], [1, 3, 4, 6]])
    )
    torch.testing.assert_close(h2, fresh2, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(h4, fresh4, rtol=0.0, atol=1e-12)
