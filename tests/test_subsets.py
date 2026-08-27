import math

import pytest
import torch

from dthoi.subsets import (
    canonicalize_subsets,
    iter_subset_batches,
    leave_one_out_subsets,
    mask_to_subset,
    subset_to_mask,
)


def test_canonicalize_subsets_sorts_rows():
    subsets = torch.tensor([[3, 1, 2], [4, 0, 2]])
    result = canonicalize_subsets(subsets, n_variables=5)
    assert torch.equal(result, torch.tensor([[1, 2, 3], [0, 2, 4]]))


def test_canonicalize_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_subsets(torch.tensor([[0, 1, 1]]))


def test_bitmask_roundtrip_is_order_independent():
    m1 = subset_to_mask((0, 4, 8, 65))
    m2 = subset_to_mask((65, 8, 4, 0))
    assert m1 == m2
    assert mask_to_subset(m1) == (0, 4, 8, 65)


def test_leave_one_out_subsets_preserves_expected_order():
    subsets = torch.tensor([[0, 2, 4], [1, 3, 5]])
    result = leave_one_out_subsets(subsets)
    expected = torch.tensor([
        [2, 4],
        [0, 4],
        [0, 2],
        [3, 5],
        [1, 5],
        [1, 3],
    ])
    assert torch.equal(result, expected)


def test_lazy_subset_batches_cover_space_once():
    N, K, batch_size = 7, 3, 5
    batches = list(iter_subset_batches(N, K, batch_size))
    subsets = torch.cat(batches, dim=0)
    assert subsets.shape[0] == math.comb(N, K)
    assert len({tuple(row.tolist()) for row in subsets}) == math.comb(N, K)
    assert all(batch.shape[0] <= batch_size for batch in batches)
