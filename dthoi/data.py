from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch


@dataclass(frozen=True)
class DiscreteData:
    """Canonical collection of discrete datasets.

    Each dataset is stored as a contiguous ``torch.uint8`` tensor with shape
    ``[T_d, N]``. Datasets may contain different numbers of observations
    ``T_d``, but they must share the same number of variables ``N`` and reside
    on the same device.

    Parameters
    ----------
    datasets
        Canonical datasets stored as a tuple of tensors.
    n_variables
        Shared number of variables across datasets.
    """

    datasets: tuple[torch.Tensor, ...]
    n_variables: int

    @property
    def n_datasets(self) -> int:
        """Number of datasets in the collection."""
        return len(self.datasets)

    @property
    def sample_counts(self) -> tuple[int, ...]:
        """Number of observations in each dataset."""
        return tuple(int(dataset.shape[0]) for dataset in self.datasets)

    @property
    def device(self) -> torch.device:
        """Device shared by all datasets."""
        return self.datasets[0].device

    def to(self, device: torch.device | str) -> DiscreteData:
        """Return this collection on ``device``.

        The existing object is returned when all datasets already reside on the
        requested device; otherwise a new canonical collection is created.
        """
        target = torch.device(device)
        if all(dataset.device == target for dataset in self.datasets):
            return self
        return DiscreteData(
            tuple(dataset.to(target) for dataset in self.datasets),
            self.n_variables,
        )


DiscreteInput: TypeAlias = DiscreteData | torch.Tensor | Sequence[torch.Tensor]


def _as_dataset_list(X: object) -> list[torch.Tensor]:
    """Convert array-like input into a list of 2-D Torch tensors."""
    if isinstance(X, torch.Tensor):
        tensor = X
    else:
        try:
            tensor = torch.as_tensor(X)
        except (TypeError, ValueError):
            tensor = None

    if tensor is not None:
        if tensor.ndim == 2:
            return [tensor]
        if tensor.ndim == 3:
            return [tensor[d] for d in range(tensor.shape[0])]
        raise ValueError("Input must have shape [T, N] or [D, T, N].")

    if not isinstance(X, Sequence) or len(X) == 0:
        raise ValueError("Input must contain at least one dataset.")

    datasets = [torch.as_tensor(item) for item in X]
    if not datasets:
        raise ValueError("Input must contain at least one dataset.")
    return datasets


def prepare_discrete_data(
    X: DiscreteInput,
    *,
    device: torch.device | str | None = None,
    validate_binary: bool = True,
) -> DiscreteData:
    """Normalize discrete observations into the canonical Torch representation.

    This function is used internally by the high-level dTHOI API. Most users
    should call :func:`dthoi.prepare_data` instead.

    Parameters
    ----------
    X
        A prepared collection, a single ``[T, N]`` tensor, a stacked
        ``[D, T, N]`` tensor, or a sequence of ``[T_d, N]`` array-like
        datasets. Objects accepted by :func:`torch.as_tensor` are also accepted
        at runtime.
    device
        Destination device. When omitted, the device of the first dataset is
        used for the full collection.
    validate_binary
        If ``True``, require every observed value to be exactly 0 or 1.

    Returns
    -------
    DiscreteData
        Contiguous ``torch.uint8`` datasets sharing one device and variable
        dimension.

    Raises
    ------
    ValueError
        If the input is empty, has an unsupported shape, mixes variable counts,
        or contains non-binary values while validation is enabled.
    """
    if isinstance(X, DiscreteData):
        return X if device is None else X.to(device)

    datasets = _as_dataset_list(X)
    target = torch.device(device) if device is not None else datasets[0].device

    prepared: list[torch.Tensor] = []
    n_variables: int | None = None

    for index, dataset in enumerate(datasets):
        if dataset.ndim != 2:
            raise ValueError(f"Dataset {index} must have shape [T, N].")
        if dataset.shape[0] == 0 or dataset.shape[1] == 0:
            raise ValueError(f"Dataset {index} must be non-empty.")

        current_n = int(dataset.shape[1])
        if n_variables is None:
            n_variables = current_n
        elif current_n != n_variables:
            raise ValueError("All datasets must share the same number of variables.")

        if validate_binary:
            valid = torch.logical_or(dataset == 0, dataset == 1)
            if not bool(torch.all(valid)):
                raise ValueError("Binary backend requires every value to be exactly 0 or 1.")

        prepared.append(dataset.to(device=target, dtype=torch.uint8).contiguous())

    assert n_variables is not None
    return DiscreteData(tuple(prepared), n_variables)
