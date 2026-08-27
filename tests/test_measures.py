import itertools

import torch

from dthoi.measures.core import nplets_measures


def test_independent_bits_have_zero_multivariate_measures():
    rows = list(itertools.product([0, 1], repeat=4))
    X = torch.tensor(rows, dtype=torch.uint8)
    values = nplets_measures(X, torch.tensor([[0, 1, 2, 3]]), count_mode="dense")
    expected = torch.zeros((1, 1, 4), dtype=torch.float64)
    assert torch.allclose(values, expected, atol=1e-12)


def test_redundant_giant_bit_matches_analytic_values():
    k = 5
    X = torch.tensor([[0] * k, [1] * k], dtype=torch.uint8)
    values = nplets_measures(X, torch.tensor([list(range(k))]), count_mode="dense")[0, 0]

    tc, dtc, o, s = values.tolist()
    assert abs(tc - (k - 1)) < 1e-12
    assert abs(dtc - 1.0) < 1e-12
    assert abs(o - (k - 2)) < 1e-12
    assert abs(s - k) < 1e-12


def test_parity_system_matches_analytic_values():
    k = 5
    rows = []
    for prefix in itertools.product([0, 1], repeat=k - 1):
        parity = sum(prefix) % 2
        rows.append((*prefix, parity))
    X = torch.tensor(rows, dtype=torch.uint8)

    values = nplets_measures(X, torch.tensor([list(range(k))]), count_mode="dense")[0, 0]
    tc, dtc, o, s = values.tolist()
    assert abs(tc - 1.0) < 1e-12
    assert abs(dtc - (k - 1)) < 1e-12
    assert abs(o - (2 - k)) < 1e-12
    assert abs(s - k) < 1e-12


def _brute_entropy(X: torch.Tensor, subset: tuple[int, ...]) -> float:
    selected = [tuple(int(v) for v in row[list(subset)].tolist()) for row in X]
    counts = {}
    for state in selected:
        counts[state] = counts.get(state, 0) + 1
    T = len(selected)
    out = 0.0
    for count in counts.values():
        p = count / T
        out -= p * __import__("math").log2(p)
    return out


def _brute_measures(X: torch.Tensor, subset: tuple[int, ...]):
    k = len(subset)
    h_joint = _brute_entropy(X, subset)
    h_single = sum(_brute_entropy(X, (i,)) for i in subset)
    h_loo = sum(_brute_entropy(X, subset[:j] + subset[j + 1 :]) for j in range(k))
    tc = h_single - h_joint
    dtc = h_loo - (k - 1) * h_joint
    return tc, dtc, tc - dtc, tc + dtc


def test_against_independent_bruteforce_calculation():
    X = torch.tensor([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=torch.uint8)
    subset = (0, 1, 2, 3)
    expected = torch.tensor(_brute_measures(X, subset), dtype=torch.float64)
    actual = nplets_measures(X, torch.tensor([subset]), count_mode="sparse")[0, 0]
    assert torch.allclose(actual, expected, atol=1e-12)


def test_multiple_datasets_preserve_dataset_axis():
    independent = torch.tensor(list(itertools.product([0, 1], repeat=3)), dtype=torch.uint8)
    redundant = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.uint8)

    values = nplets_measures(
        [independent, redundant],
        torch.tensor([[0, 1, 2]]),
        count_mode="dense",
    )

    assert values.shape == (1, 2, 4)
    assert torch.allclose(values[0, 0], torch.zeros(4, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(
        values[0, 1],
        torch.tensor([2.0, 1.0, 1.0, 3.0], dtype=torch.float64),
        atol=1e-12,
    )
