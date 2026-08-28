"""Shared definitions for higher-order information measures."""

from __future__ import annotations

from typing import TypeAlias

import torch

LocalMeasureValues: TypeAlias = tuple[torch.Tensor, ...]
MeasureOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, LocalMeasureValues]


def finish_measure_channels(values: torch.Tensor) -> None:
    """Fill O-information and S-information from TC and DTC in place."""
    values[..., 2] = values[..., 0] - values[..., 1]
    values[..., 3] = values[..., 0] + values[..., 1]


def combine_global_terms(
    h_joint: torch.Tensor,
    h_single: torch.Tensor,
    h_loo: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """Combine shared entropy terms into TC, DTC, O-information, and S-information.

    Parameters
    ----------
    h_joint
        Entropy of each complete variable set, in bits.
    h_single
        Sum of the singleton entropies for each complete variable set, in bits.
    h_loo
        Sum of the leave-one-out entropies for each complete variable set, in bits.
    order
        Number of variables in each complete variable set.

    Returns
    -------
    torch.Tensor
        Tensor whose final dimension contains TC, DTC, O-information, and
        S-information, in that order and in bits.

    Notes
    -----
    dTHOI uses ``O = TC - DTC`` and ``S = TC + DTC`` throughout the package.
    The leading dimensions of the three entropy arguments may represent
    variable sets, datasets, posterior-grid points, or any compatible batch.
    """
    values = torch.empty((*h_joint.shape, 4), dtype=torch.float64, device=h_joint.device)
    values[..., 0] = h_single - h_joint
    values[..., 1] = h_loo - (order - 1) * h_joint
    finish_measure_channels(values)
    return values
