import pytest
import torch

from dthoi.data import prepare_discrete_data


def test_prepare_single_dataset():
    X = torch.tensor([[0, 1], [1, 0], [1, 1]])
    data = prepare_discrete_data(X)
    assert data.n_datasets == 1
    assert data.n_variables == 2
    assert data.sample_counts == (3,)
    assert data.datasets[0].dtype == torch.uint8
    assert data.datasets[0].is_contiguous()


def test_prepare_stacked_datasets():
    X = torch.tensor([
        [[0, 1], [1, 0]],
        [[1, 1], [0, 0]],
    ])
    data = prepare_discrete_data(X)
    assert data.n_datasets == 2
    assert data.sample_counts == (2, 2)


def test_prepare_ragged_datasets():
    X = [
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([[0, 0], [1, 1], [1, 0]]),
    ]
    data = prepare_discrete_data(X)
    assert data.sample_counts == (2, 3)


def test_prepare_reuses_already_prepared_data():
    data = prepare_discrete_data(torch.tensor([[0, 1], [1, 0]]))
    assert prepare_discrete_data(data) is data
    assert data.to("cpu") is data


def test_reject_nonbinary_values():
    X = torch.tensor([[0, 2], [1, 0]])
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        prepare_discrete_data(X)


def test_reject_inconsistent_variable_counts():
    X = [torch.zeros((4, 2)), torch.zeros((4, 3))]
    with pytest.raises(ValueError, match="same number of variables"):
        prepare_discrete_data(X)
