from __future__ import annotations

import torch

from ..data import DiscreteInput, PreparedDiscreteData, prepare_discrete_data
from ..subsets import _masks_from_canonical_subsets, canonicalize_subsets
from .cache import EntropyCache
from .counting import (
    CountMode,
    choose_count_mode,
    dense_counts_from_codes,
    encode_binary_rows,
    encode_binary_states,
    mask_binary_rows,
    sparse_counts_from_codes,
)
from .estimators import CountEntropyEstimator, resolve_estimator


class CountingEntropyProvider:
    """Exact-counting entropy backend for binary discrete observations.

    Parameters
    ----------
    X
        Discrete observations accepted by :func:`prepare_discrete_data`.
    estimator
        Entropy estimator name or estimator instance.
    count_mode
        ``"dense"``, ``"sparse"``, or ``"auto"`` counting strategy.
    cache
        Optional entropy cache shared across calls.
    device
        Destination device for the prepared observations.
    dense_state_limit
        Maximum state-space cardinality considered by automatic dense counting.
    dense_memory_limit_bytes
        Maximum estimated dense working memory considered by automatic counting.
    """

    def __init__(
        self,
        X: DiscreteInput,
        *,
        estimator: str | CountEntropyEstimator = "plugin",
        count_mode: CountMode = "auto",
        cache: EntropyCache | None = None,
        device: torch.device | str | None = None,
        dense_state_limit: int = 1 << 16,
        dense_memory_limit_bytes: int = 256 * 1024 * 1024,
    ):
        """Prepare data, estimator, counting policy, and cache state."""
        self.data: PreparedDiscreteData = prepare_discrete_data(X, device=device)
        self.estimator = resolve_estimator(estimator)
        self.count_mode = count_mode
        self.cache = cache if cache is not None else EntropyCache()
        self.dense_state_limit = dense_state_limit
        self.dense_memory_limit_bytes = dense_memory_limit_bytes
        self._row_codes: list[torch.Tensor | None] = [None] * self.data.n_datasets

    def entropy(self, subsets: torch.Tensor) -> torch.Tensor:
        """Estimate entropy for a fixed-order batch of variable subsets.

        Parameters
        ----------
        subsets
            Integer subset indices with shape ``[B, K]`` or ``[K]``.

        Returns
        -------
        torch.Tensor
            Entropies in bits with shape ``[B, D]``, where ``D`` is the number
            of prepared datasets.
        """
        subsets = canonicalize_subsets(subsets, self.data.n_variables)
        masks = _masks_from_canonical_subsets(subsets)
        batch_size, order = subsets.shape
        n_datasets = self.data.n_datasets
        device = self.data.device

        result = torch.empty((batch_size, n_datasets), dtype=torch.float64, device=device)

        missing: dict[int, torch.Tensor] = {}
        positions: dict[int, list[int]] = {}

        for position, (mask, subset) in enumerate(zip(masks, subsets, strict=True)):
            positions.setdefault(mask, []).append(position)
            cached = self.cache.get(mask)
            if cached is not None:
                result[position] = cached.to(device=device, dtype=torch.float64)
            elif mask not in missing:
                missing[mask] = subset

        if not missing:
            return result

        missing_masks = list(missing)
        missing_subsets = torch.stack(list(missing.values())).to(device)
        n_missing = len(missing_masks)
        entropy_values = torch.empty(
            (n_missing, n_datasets), dtype=torch.float64, device=device
        )

        for dataset_index, dataset in enumerate(self.data.datasets):
            mode = choose_count_mode(
                order=order,
                n_subsets=n_missing,
                n_samples=dataset.shape[0],
                requested=self.count_mode,
                dense_state_limit=self.dense_state_limit,
                dense_memory_limit_bytes=self.dense_memory_limit_bytes,
            )

            if mode == "dense":
                codes = encode_binary_states(dataset, missing_subsets)
                counts = dense_counts_from_codes(codes, 1 << order)
                values = self.estimator.entropy_from_dense(counts)
            else:
                if self.data.n_variables <= 63:
                    row_codes = self._row_codes[dataset_index]
                    if row_codes is None:
                        row_codes = encode_binary_rows(dataset)
                        self._row_codes[dataset_index] = row_codes
                    codes = mask_binary_rows(row_codes, missing_masks)
                else:
                    codes = encode_binary_states(dataset, missing_subsets)
                counts = sparse_counts_from_codes(codes)
                values = self.estimator.entropy_from_sparse(counts)

            entropy_values[:, dataset_index] = values.to(device)

        for row, mask in enumerate(missing_masks):
            value = entropy_values[row]
            self.cache.put(mask, value)
            for position in positions[mask]:
                result[position] = value

        return result
