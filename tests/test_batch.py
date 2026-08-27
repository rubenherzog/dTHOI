import math

import torch

from dthoi.batch import multi_order_measures
from dthoi.measures.core import nplets_measures


def test_multi_order_returns_all_subsets_without_materializing_global_space():
    X = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
    ], dtype=torch.uint8)

    batches = multi_order_measures(X, min_order=2, max_order=4, batch_size=3)
    total = sum(subsets.shape[0] for subsets, _ in batches)
    expected = sum(math.comb(5, k) for k in range(2, 5))
    assert total == expected
    assert all(subsets.device.type == "cpu" and values.device.type == "cpu" for subsets, values in batches)


def test_multi_order_matches_direct_scoring_for_one_subset():
    X = torch.tensor([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ], dtype=torch.uint8)

    target = torch.tensor([[0, 1, 3]])
    direct = nplets_measures(X, target, count_mode="dense")[0]

    batches = multi_order_measures(X, min_order=3, max_order=3, batch_size=2, count_mode="dense")
    found = None
    for subsets, values in batches:
        for i, subset in enumerate(subsets):
            if torch.equal(subset, target[0]):
                found = values[i]
                break
    assert found is not None
    assert torch.allclose(found, direct, atol=1e-12)
