import torch

import dthoi


def _example_data() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=torch.uint8,
    )


def test_public_api_uses_scientific_names():
    expected = {
        "DiscreteData",
        "InformationMeasures",
        "InteractionResults",
        "analyze_orders",
        "estimate_entropy",
        "information_measures",
        "prepare_data",
    }
    assert set(dthoi.__all__) == expected
    assert not hasattr(dthoi, "CountingEntropyProvider")
    assert not hasattr(dthoi, "EntropyCache")
    assert not hasattr(dthoi, "nplets_measures")
    assert not hasattr(dthoi, "multi_order_measures")


def test_prepare_data_returns_reusable_discrete_data():
    prepared = dthoi.prepare_data(_example_data())
    assert isinstance(prepared, dthoi.DiscreteData)
    assert type(prepared).__name__ == "DiscreteData"
    assert prepared.n_variables == 4
    assert prepared.sample_counts == (5,)


def test_entropy_accepts_plain_variable_sets():
    values = dthoi.estimate_entropy(_example_data(), [[0, 1], [1, 2]])
    assert values.shape == (2, 1)
    assert values.dtype == torch.float64


def test_information_measures_have_named_accessors():
    results = dthoi.information_measures(_example_data(), [[0, 1, 2], [1, 2, 3]])

    assert isinstance(results, dthoi.InformationMeasures)
    assert results.values.shape == (2, 1, 4)
    torch.testing.assert_close(results.total_correlation, results.values[..., 0])
    torch.testing.assert_close(results.dual_total_correlation, results.values[..., 1])
    torch.testing.assert_close(results.o_information, results.values[..., 2])
    torch.testing.assert_close(results.s_information, results.values[..., 3])
    assert set(results.as_dict()) == {
        "total_correlation",
        "dual_total_correlation",
        "o_information",
        "s_information",
    }


def test_analyze_orders_returns_named_results():
    results = dthoi.analyze_orders(
        _example_data(),
        min_order=3,
        max_order=3,
        sets_per_batch=2,
    )

    assert results
    assert all(isinstance(result, dthoi.InteractionResults) for result in results)
    assert all(result.order == 3 for result in results)
    assert sum(result.variable_sets.shape[0] for result in results) == 4
    assert all(result.information.values.shape[-1] == 4 for result in results)


def test_prepared_data_can_be_reused_across_public_functions():
    prepared = dthoi.prepare_data(_example_data())
    variable_sets = [[0, 1, 2], [1, 2, 3]]

    direct_entropy = dthoi.estimate_entropy(_example_data(), variable_sets)
    prepared_entropy = dthoi.estimate_entropy(prepared, variable_sets)
    direct_information = dthoi.information_measures(_example_data(), variable_sets).values
    prepared_information = dthoi.information_measures(prepared, variable_sets).values

    torch.testing.assert_close(direct_entropy, prepared_entropy, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(direct_information, prepared_information, rtol=0.0, atol=1e-12)
