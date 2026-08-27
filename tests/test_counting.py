import torch

from dthoi.entropy.base import CountingEntropyProvider
from dthoi.entropy.counting import choose_count_mode


def _binary_data(T: int, N: int, seed: int) -> torch.Tensor:
    return torch.randint(
        0,
        2,
        (T, N),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(seed),
    )


def test_sparse_masked_encoding_matches_dense_at_single_word_boundary():
    X = _binary_data(257, 63, 20)
    subsets = torch.tensor([
        [0, 17, 31, 62],
        [1, 16, 32, 61],
        [3, 9, 27, 58],
    ])
    dense = CountingEntropyProvider(X, count_mode="dense").entropy(subsets)
    sparse = CountingEntropyProvider(X, count_mode="sparse").entropy(subsets)
    torch.testing.assert_close(dense, sparse, rtol=0.0, atol=1e-12)


def test_sparse_fallback_matches_dense_above_single_word_dataset_width():
    X = _binary_data(257, 64, 21)
    subsets = torch.tensor([
        [0, 17, 31, 63],
        [1, 16, 32, 62],
        [3, 9, 27, 58],
    ])
    dense = CountingEntropyProvider(X, count_mode="dense").entropy(subsets)
    sparse = CountingEntropyProvider(X, count_mode="sparse").entropy(subsets)
    torch.testing.assert_close(dense, sparse, rtol=0.0, atol=1e-12)


def test_auto_count_mode_budgets_entropy_working_memory():
    mode = choose_count_mode(
        order=14,
        n_subsets=1920,
        n_samples=5000,
        requested="auto",
        dense_state_limit=1 << 16,
        dense_memory_limit_bytes=256 * 1024 * 1024,
    )
    assert mode == "sparse"


def test_auto_count_mode_keeps_small_dense_workloads_dense():
    mode = choose_count_mode(
        order=7,
        n_subsets=4096,
        n_samples=1000,
        requested="auto",
        dense_state_limit=1 << 16,
        dense_memory_limit_bytes=256 * 1024 * 1024,
    )
    assert mode == "dense"
