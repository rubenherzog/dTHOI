from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

ArrayLike2D = np.ndarray | torch.Tensor


@dataclass(frozen=True)
class PreparedDiscreteData:
    """Canonical collection of discrete datasets.

    Each dataset is stored as a contiguous tensor of shape ``[T_d, N]``.
    Different datasets may have different sample counts ``T_d`` but must share
    the same number of variables ``N``.
    """

    datasets: tuple[torch.Tensor, ...]
    n_variables: int

    @property
    def n_datasets(self) -> int:
        return len(self.datasets)

    @property
    def sample_counts(self) -> tuple[int, ...]:
        return tuple(int(x.shape[0]) for x in self.datasets)

    @property
    def device(self) -> torch.device:
        return self.datasets[0].device


def _as_dataset_list(X: ArrayLike2D | Sequence[ArrayLike2D]) -> list[torch.Tensor]:
    if isinstance(X, torch.Tensor):
        if X.ndim == 2:
            return [X]
        if X.ndim == 3:
            return [X[d] for d in range(X.shape[0])]
        raise ValueError("Tensor input must have shape [T, N] or [D, T, N].")

    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            return [torch.as_tensor(X)]
        if X.ndim == 3:
            return [torch.as_tensor(X[d]) for d in range(X.shape[0])]
        raise ValueError("NumPy input must have shape [T, N] or [D, T, N].")

    if not isinstance(X, Sequence) or len(X) == 0:
        raise ValueError("Input must contain at least one dataset.")

    return [torch.as_tensor(item) for item in X]


def prepare_discrete_data(
    X: ArrayLike2D | Sequence[ArrayLike2D],
    *,
    device: torch.device | str | None = None,
    validate_binary: bool = True,
) -> PreparedDiscreteData:
    """Normalize discrete data without changing its statistical content.

    Parameters
    ----------
    X
        A single ``[T, N]`` dataset, a stacked ``[D, T, N]`` tensor/array, or
        a sequence of ``[T_d, N]`` datasets.
    device
        Destination device. If omitted, tensors remain on their existing device
        when possible.
    validate_binary
        Require values to be exactly 0 or 1. Binary variables are the initial
        supported backend; the container does not otherwise encode this as a
        permanent architectural restriction.
    """

    datasets = _as_dataset_list(X)
    target = torch.device(device) if device is not None else None

    prepared: list[torch.Tensor] = []
    n_variables: int | None = None

    for i, data in enumerate(datasets):
        if data.ndim != 2:
            raise ValueError(f"Dataset {i} must have shape [T, N].")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f"Dataset {i} must be non-empty.")

        if n_variables is None:
            n_variables = int(data.shape[1])
        elif int(data.shape[1]) != n_variables:
            raise ValueError("All datasets must share the same number of variables.")

        if validate_binary:
            valid = torch.logical_or(data == 0, data == 1)
            if not bool(torch.all(valid)):
                raise ValueError("Binary backend requires every value to be exactly 0 or 1.")

        if target is not None:
            data = data.to(target)
        data = data.to(dtype=torch.uint8).contiguous()
        prepared.append(data)

    assert n_variables is not None
    return PreparedDiscreteData(tuple(prepared), n_variables)
