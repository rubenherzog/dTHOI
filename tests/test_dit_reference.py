import itertools

import pytest
import torch

from dthoi.measures.core import nplets_measures


dit = pytest.importorskip("dit")


def _distribution_from_exact_samples(X: torch.Tensor):
    outcomes = [tuple(int(v) for v in row.tolist()) for row in X]
    unique = sorted(set(outcomes))
    counts = [outcomes.count(o) for o in unique]
    total = sum(counts)
    pmf = [c / total for c in counts]
    return dit.Distribution(unique, pmf)


@pytest.mark.parametrize("kind", ["independent", "redundant", "parity"])
def test_core_measures_match_dit(kind):
    k = 4
    if kind == "independent":
        X = torch.tensor(list(itertools.product([0, 1], repeat=k)), dtype=torch.uint8)
    elif kind == "redundant":
        X = torch.tensor([[0] * k, [1] * k], dtype=torch.uint8)
    else:
        rows = []
        for prefix in itertools.product([0, 1], repeat=k - 1):
            rows.append((*prefix, sum(prefix) % 2))
        X = torch.tensor(rows, dtype=torch.uint8)

    dist = _distribution_from_exact_samples(X)
    expected = torch.tensor(
        [
            dit.multivariate.total_correlation(dist),
            dit.multivariate.dual_total_correlation(dist),
            dit.multivariate.o_information(dist),
            dit.multivariate.s_information(dist),
        ],
        dtype=torch.float64,
    )
    actual = nplets_measures(
        X,
        torch.tensor([list(range(k))]),
        count_mode="dense",
    )[0, 0]

    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=1e-12)
