import pytest
import torch

import dthoi
from dthoi.entropy.base import CountingEntropyProvider
from dthoi.measures.core import measures_from_provider

ENTROPY_ESTIMATORS = (
    "empirical", "miller_madow", "schurmann", "shrinkage",
    "chao_shen", "pitman_yor", "ansb",
)
COHERENT_ESTIMATORS = ("dirichlet_a1", "dirichlet_eb", "nsb")


def _repeated_binary_data() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260828)
    base = torch.randint(0, 2, (64, 6), dtype=torch.uint8, generator=generator)
    return base.repeat((8, 1))


@pytest.mark.parametrize("estimator", ENTROPY_ESTIMATORS)
def test_dispatch_preserves_existing_entropy_measure_path(estimator: str):
    data = _repeated_binary_data()
    variable_sets = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=torch.long)
    provider = CountingEntropyProvider(data, estimator=estimator, count_mode="auto")
    expected = measures_from_provider(provider, variable_sets)
    actual = dthoi.information_measures(
        data, variable_sets, estimator=estimator
    ).values
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("estimator", COHERENT_ESTIMATORS)
def test_coherent_batched_results_match_setwise_execution(estimator: str):
    data = _repeated_binary_data()
    variable_sets = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long
    )
    batched = dthoi.information_measures(
        data, variable_sets, estimator=estimator
    ).values
    setwise = torch.cat(
        [
            dthoi.information_measures(
                data, variable_set, estimator=estimator
            ).values
            for variable_set in variable_sets
        ],
        dim=0,
    )
    torch.testing.assert_close(batched, setwise, rtol=1e-12, atol=1e-12)
