import math
import warnings

import pytest
import torch

import dthoi
from dthoi.entropy.base import CountingEntropyProvider
from dthoi.entropy.counting import dense_counts_from_codes, sparse_counts_from_codes
from dthoi.entropy.estimators import (
    AsymptoticNsbEstimator,
    ChaoShenEstimator,
    PitmanYorEstimator,
    SchurmannEstimator,
    ShrinkageEstimator,
    resolve_estimator,
)


REFERENCE_COUNTS = torch.tensor(
    [[4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=torch.int64,
)


def _balanced_four_variable_data() -> torch.Tensor:
    """Return every four-bit state twice for well-behaved marginal counts."""
    rows = [
        [(state >> bit) & 1 for bit in range(4)]
        for state in range(16)
    ]
    return torch.tensor(rows + rows, dtype=torch.uint8)


def _local_four_variable_data() -> torch.Tensor:
    """Return repeated and singleton states for corrected local-value tests."""
    return torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=torch.uint8,
    )


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (SchurmannEstimator(), 2.095761233437094),
        (ShrinkageEstimator(), 3.059223453573027),
        (ChaoShenEstimator(), 2.277111342708328),
        (PitmanYorEstimator(), 2.723065014540778),
        (AsymptoticNsbEstimator(max_sample_to_state_ratio=1.0), 4.020551446257301),
    ],
)
def test_estimators_match_fixed_scalar_reference_values(estimator, expected):
    """Validate each vectorized estimator against its scalar defining equation."""
    actual = estimator.entropy_from_dense(REFERENCE_COUNTS, n_states=16)
    torch.testing.assert_close(
        actual,
        torch.tensor([expected], dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_schurmann_matches_dit_entropy2_published_test_case():
    """Match the reference value used by dit for counts seven and three."""
    counts = torch.tensor([[7, 3]], dtype=torch.int64)
    value = SchurmannEstimator().entropy_from_dense(counts, n_states=2)
    assert value.item() == pytest.approx(1.1187360918572902, abs=1e-12)


def test_all_new_estimators_are_batched_and_dense_sparse_equivalent():
    """Require one vectorized result per histogram with dense/sparse parity."""
    codes = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 2, 3],
            [0, 0, 1, 1, 1, 2, 2, 3],
            [0, 1, 1, 2, 2, 2, 3, 3],
        ],
        dtype=torch.int64,
    )
    dense = dense_counts_from_codes(codes, n_states=16)
    sparse = sparse_counts_from_codes(codes)
    estimators = [
        SchurmannEstimator(),
        ShrinkageEstimator(),
        ChaoShenEstimator(),
        PitmanYorEstimator(),
        AsymptoticNsbEstimator(max_sample_to_state_ratio=1.0),
    ]

    for estimator in estimators:
        dense_values = estimator.entropy_from_dense(dense, n_states=16)
        sparse_values = estimator.entropy_from_sparse(sparse, n_states=16)
        assert dense_values.shape == (3,)
        torch.testing.assert_close(dense_values, sparse_values, rtol=0.0, atol=1e-12)


def test_schurmann_local_values_match_state_additive_formula():
    """Validate Schürmann local values independently from the global reduction."""
    counts = torch.tensor([[3, 1]], dtype=torch.int64)
    codes = torch.tensor([[0, 0, 0, 1]], dtype=torch.int64)
    estimator = SchurmannEstimator()
    local = estimator.local_from_dense(counts, codes)

    total = torch.tensor(4.0, dtype=torch.float64)
    expected_by_count = []
    for count in (3.0, 1.0):
        n = torch.tensor(count, dtype=torch.float64)
        integral = 0.5 * (
            torch.special.digamma((n + 1.0) / 2.0)
            - torch.special.digamma(n / 2.0)
        )
        signed = integral if int(count) % 2 else -integral
        expected_by_count.append(
            (
                torch.special.digamma(total)
                - torch.special.digamma(n)
                + signed
            )
            / math.log(2.0)
        )
    expected = torch.stack(
        [expected_by_count[0], expected_by_count[0], expected_by_count[0], expected_by_count[1]]
    ).unsqueeze(0)
    torch.testing.assert_close(local, expected, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        local.mean(dim=1),
        estimator.entropy_from_dense(counts, n_states=2),
        rtol=0.0,
        atol=1e-12,
    )


def test_chao_shen_local_values_match_horvitz_thompson_formula():
    """Validate Chao-Shen observation contributions from the published estimator."""
    counts = torch.tensor([[3, 1]], dtype=torch.int64)
    codes = torch.tensor([[0, 0, 0, 1]], dtype=torch.int64)
    estimator = ChaoShenEstimator()
    local = estimator.local_from_dense(counts, codes)

    total = 4.0
    coverage = 1.0 - 1.0 / total
    expected_by_count = []
    for count in (3.0, 1.0):
        adjusted = coverage * count / total
        detection = 1.0 - (1.0 - adjusted) ** total
        expected_by_count.append(
            -coverage * math.log2(adjusted) / detection
        )
    expected = torch.tensor(
        [[expected_by_count[0], expected_by_count[0], expected_by_count[0], expected_by_count[1]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(local, expected, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        local.mean(dim=1),
        estimator.entropy_from_dense(counts, n_states=2),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("name", ["schurmann", "chao_shen"])
def test_supported_corrected_local_entropy_is_batched_dense_sparse_and_mean_consistent(name):
    """Preserve batch rows and exact global means in both counting paths."""
    data = _local_four_variable_data()
    subsets = torch.tensor([[0, 1, 2], [1, 2, 3]])

    dense_provider = CountingEntropyProvider(data, estimator=name, count_mode="dense")
    sparse_provider = CountingEntropyProvider(data, estimator=name, count_mode="sparse")
    dense_entropy, dense_local = dense_provider.entropy_with_local(subsets)
    sparse_entropy, sparse_local = sparse_provider.entropy_with_local(subsets)

    assert dense_local[0].shape == (2, data.shape[0])
    assert sparse_local[0].shape == (2, data.shape[0])
    torch.testing.assert_close(dense_entropy, sparse_entropy, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(dense_local[0], sparse_local[0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(
        dense_local[0].mean(dim=1), dense_entropy[:, 0], rtol=0.0, atol=1e-12
    )


def test_shrinkage_includes_unobserved_nominal_states_analytically():
    """Changing Q changes shrinkage without materializing zero-count states."""
    counts = torch.tensor([[4, 4]], dtype=torch.int64)
    estimator = ShrinkageEstimator()

    observed_support = estimator.entropy_from_dense(counts, n_states=2)
    larger_support = estimator.entropy_from_dense(counts, n_states=4)

    torch.testing.assert_close(
        observed_support,
        torch.tensor([1.0], dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert larger_support.item() > observed_support.item()


def test_chao_shen_is_undefined_when_every_observation_is_a_singleton():
    """Do not replace zero estimated coverage with an ad-hoc fallback."""
    counts = torch.ones((1, 8), dtype=torch.int64)
    codes = torch.arange(8, dtype=torch.int64).unsqueeze(0)
    estimator = ChaoShenEstimator()
    value = estimator.entropy_from_dense(counts, n_states=8)
    local = estimator.local_from_dense(counts, codes)
    assert torch.isnan(value).all()
    assert torch.isnan(local).all()


def test_ansb_separates_observed_coincidences_from_nominal_support():
    """Use K1 for Delta while reserving Q for the undersampling diagnostic."""
    counts = torch.tensor([[3, 2, 1, 0, 0, 0, 0, 0]], dtype=torch.int64)
    estimator = AsymptoticNsbEstimator(max_sample_to_state_ratio=1.0)
    value = estimator.entropy_from_dense(counts, n_states=8)

    sample_count = 6.0
    observed_states = 3.0
    delta = sample_count - observed_states
    expected = (
        0.5772156649015329
        - math.log(2.0)
        + 2.0 * math.log(sample_count)
        - torch.special.digamma(torch.tensor(delta, dtype=torch.float64)).item()
    ) / math.log(2.0)
    assert value.item() == pytest.approx(expected, abs=1e-12)


def test_ansb_warns_outside_regime_and_is_nan_without_coincidences():
    """Make both asymptotic limitations observable to scientific users."""
    estimator = AsymptoticNsbEstimator()
    well_sampled = torch.tensor([[4, 4, 0, 0]], dtype=torch.int64)
    with pytest.warns(RuntimeWarning, match="severe undersampling"):
        estimator.entropy_from_dense(well_sampled, n_states=4)

    all_singletons = torch.ones((1, 16), dtype=torch.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = estimator.entropy_from_dense(all_singletons, n_states=16)
    assert torch.isnan(value).all()


def test_pitman_yor_hyperparameter_domain_is_explicit():
    """Reject parameter values outside the supported infinite-PY regime."""
    with pytest.raises(ValueError, match="discount"):
        PitmanYorEstimator(discount=1.0)
    with pytest.raises(ValueError, match="concentration"):
        PitmanYorEstimator(discount=0.0, concentration=0.0)


def test_resolver_exposes_the_five_new_scientific_names():
    """Keep public estimator selection simple while retaining internal classes."""
    assert isinstance(resolve_estimator("schurmann"), SchurmannEstimator)
    assert isinstance(resolve_estimator("shrinkage"), ShrinkageEstimator)
    assert isinstance(resolve_estimator("chao_shen"), ChaoShenEstimator)
    assert isinstance(resolve_estimator("pitman_yor"), PitmanYorEstimator)
    assert isinstance(resolve_estimator("ansb"), AsymptoticNsbEstimator)


def test_only_estimators_without_canonical_observation_decomposition_reject_local_values():
    """Reject local requests that would require arbitrary redistribution."""
    data = _local_four_variable_data()[:, :3]
    for name in ("shrinkage", "pitman_yor", "ansb"):
        provider = CountingEntropyProvider(data, estimator=name, count_mode="sparse")
        with pytest.raises(NotImplementedError, match="does not define local values"):
            provider.entropy_with_local(torch.tensor([[0, 1, 2]]))


def test_public_entropy_api_preserves_batch_and_dataset_axes():
    """Keep the scientific API shape contract for all five estimators."""
    first = _local_four_variable_data()
    second = first[:6]
    variable_sets = [[0, 1, 2], [1, 2, 3]]

    for name in ("schurmann", "shrinkage", "chao_shen", "pitman_yor"):
        values = dthoi.estimate_entropy(
            [first, second], variable_sets, entropy_estimator=name
        )
        assert values.shape == (2, 2)
        assert values.dtype == torch.float64

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = dthoi.estimate_entropy(
            [first, second], variable_sets, entropy_estimator="ansb"
        )
    assert values.shape == (2, 2)
    assert values.dtype == torch.float64


def test_new_estimators_flow_through_shared_information_measure_definitions():
    """Derive all four measures from shared entropies for every new estimator."""
    data = _balanced_four_variable_data()
    variable_sets = [[0, 1, 2], [1, 2, 3]]

    for name in ("schurmann", "shrinkage", "chao_shen", "pitman_yor", "ansb"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = dthoi.information_measures(
                data, variable_sets, entropy_estimator=name
            )
        assert result.values.shape == (2, 1, 4)
        assert torch.isfinite(result.values).all()
        torch.testing.assert_close(
            result.o_information,
            result.total_correlation - result.dual_total_correlation,
            rtol=0.0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            result.s_information,
            result.total_correlation + result.dual_total_correlation,
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize("name", ["schurmann", "chao_shen"])
def test_corrected_local_information_measures_average_to_global_values(name):
    """Propagate supported corrected locals through shared TC/DTC/O/S equations."""
    data = _balanced_four_variable_data()
    result = dthoi.information_measures(
        data,
        [[0, 1, 2], [1, 2, 3]],
        entropy_estimator=name,
        local_values=True,
        max_local_values_per_batch=data.shape[0],
    )
    assert result.local is not None
    assert result.local.values[0].shape == (2, data.shape[0], 4)
    torch.testing.assert_close(
        result.local.values[0].mean(dim=1),
        result.values[:, 0, :],
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.local.o_information[0],
        result.local.total_correlation[0] - result.local.dual_total_correlation[0],
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.local.s_information[0],
        result.local.total_correlation[0] + result.local.dual_total_correlation[0],
        rtol=0.0,
        atol=1e-12,
    )


def test_analyze_orders_preserves_generated_batch_bound_with_new_estimator():
    """Keep exhaustive subset generation lazy and bounded after estimator expansion."""
    results = dthoi.analyze_orders(
        _balanced_four_variable_data(),
        min_order=3,
        max_order=3,
        entropy_estimator="shrinkage",
        sets_per_batch=2,
    )
    assert results
    assert all(item.variable_sets.shape[0] <= 2 for item in results)
    assert sum(item.variable_sets.shape[0] for item in results) == math.comb(4, 3)