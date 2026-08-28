from itertools import product

import pytest
import torch

import dthoi

ESTIMATORS = ("dirichlet_a1", "dirichlet_eb", "nsb")


def _redundant_data(order: int, repeats: int = 128) -> torch.Tensor:
    source = torch.tensor([0, 1], dtype=torch.uint8).repeat_interleave(repeats)
    return source.unsqueeze(1).expand(-1, order).contiguous()


def _parity_data(order: int, repeats: int = 32) -> torch.Tensor:
    leading = torch.tensor(list(product((0, 1), repeat=order - 1)), dtype=torch.uint8)
    parity = torch.remainder(leading.sum(dim=1, keepdim=True), 2).to(torch.uint8)
    return torch.cat((leading, parity), dim=1).repeat_interleave(repeats, dim=0)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_coherent_estimators_preserve_measure_identities_and_coherence(estimator: str):
    generator = torch.Generator().manual_seed(20260828)
    data = torch.randint(0, 2, (512, 6), dtype=torch.uint8, generator=generator)
    result = dthoi.information_measures(
        data, [[0, 1, 2, 3], [2, 3, 4, 5]], estimator=estimator
    )
    torch.testing.assert_close(
        result.o_information,
        result.total_correlation - result.dual_total_correlation,
        rtol=0.0, atol=1e-12,
    )
    torch.testing.assert_close(
        result.s_information,
        result.total_correlation + result.dual_total_correlation,
        rtol=0.0, atol=1e-12,
    )
    assert bool(torch.all(result.total_correlation >= -1e-12))
    assert bool(torch.all(result.dual_total_correlation >= -1e-12))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_coherent_estimators_recover_redundant_and_parity_signs(estimator: str):
    variable_set = torch.arange(5)
    redundant_o = dthoi.information_measures(
        _redundant_data(5), variable_set, estimator=estimator
    ).o_information.item()
    parity_o = dthoi.information_measures(
        _parity_data(5), variable_set, estimator=estimator
    ).o_information.item()
    assert redundant_o > 0.0
    assert parity_o < 0.0


def test_coherent_estimators_deduplicate_repeated_parent_sets():
    data = _redundant_data(5)
    variable_sets = torch.tensor(
        [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 4]], dtype=torch.long
    )
    result = dthoi.information_measures(
        data, variable_sets, estimator="dirichlet_eb"
    ).values
    torch.testing.assert_close(result[0], result[1], rtol=0.0, atol=0.0)
    assert result.shape == (3, 1, 4)


def test_coherent_estimators_support_wide_sets_in_500_variable_systems():
    generator = torch.Generator().manual_seed(7)
    data = torch.randint(0, 2, (24, 500), dtype=torch.uint8, generator=generator)
    variable_set = torch.linspace(0, 499, 70, dtype=torch.float64).round().to(torch.long)
    assert torch.unique(variable_set).numel() == 70
    result = dthoi.information_measures(
        data, variable_set, estimator="dirichlet_a1"
    )
    assert int(variable_set.max()) > 63
    assert result.values.shape == (1, 1, 4)
    assert bool(torch.isfinite(result.values).all())


def test_coherent_estimators_keep_unequal_datasets_separate():
    data = _parity_data(4, repeats=16)
    result = dthoi.information_measures(
        [data[:80], data[:48]], [0, 1, 2, 3], estimator="nsb"
    )
    assert result.values.shape == (1, 2, 4)
    assert bool(torch.isfinite(result.values).all())


def test_coherent_estimators_do_not_expose_local_values():
    with pytest.raises(NotImplementedError, match="does not define sample-resolved"):
        dthoi.information_measures(
            _redundant_data(4), [0, 1, 2, 3], estimator="nsb", local_values=True
        )


def test_coherent_estimators_work_through_analyze_orders():
    results = dthoi.analyze_orders(
        _parity_data(4, repeats=8), min_order=3, max_order=3,
        estimator="dirichlet_a1", sets_per_batch=2,
    )
    assert sum(item.variable_sets.shape[0] for item in results) == 4
    assert all(item.information.values.shape[-1] == 4 for item in results)


def test_entropy_api_rejects_context_dependent_coherent_estimators():
    with pytest.raises(ValueError, match="Unknown estimator"):
        dthoi.estimate_entropy(
            _redundant_data(4), [0, 1, 2, 3], entropy_estimator="nsb"
        )
