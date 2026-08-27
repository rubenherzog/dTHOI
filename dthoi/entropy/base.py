from __future__ import annotations

from collections import OrderedDict

import torch

from ..data import PreparedDiscreteData, prepare_discrete_data
from ..subsets import canonicalize_subsets, masks_from_subsets
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
    """Entropy backend for binary samples using exact state counting."""

    def __init__(
        self,
        X: PreparedDiscreteData | torch.Tensor | list[torch.Tensor],
        *,
        estimator: str | CountEntropyEstimator = "plugin",
        count_mode: CountMode = "auto",
        cache: EntropyCache | None = None,
        device: torch.device | str | None = None,
        dense_state_limit: int = 1 << 16,
        dense_memory_limit_bytes: int = 256 * 1024 * 1024,
    ):
        self.data = X if isinstance(X, PreparedDiscreteData) else prepare_discrete_data(X, device=device)
        if device is not None and isinstance(X, PreparedDiscreteData):
            target = torch.device(device)
            if any(ds.device != target for ds in X.datasets):
                self.data = PreparedDiscreteData(tuple(ds.to(target) for ds in X.datasets), X.n_variables)
        self.estimator = resolve_estimator(estimator)
        self.count_mode = count_mode
        self.cache = cache if cache is not None else EntropyCache()
        self.dense_state_limit = dense_state_limit
        self.dense_memory_limit_bytes = dense_memory_limit_bytes
        self._row_codes: list[torch.Tensor | None] = [None] * self.data.n_datasets

    def entropy(self, subsets: torch.Tensor) -> torch.Tensor:
        subsets = canonicalize_subsets(subsets, self.data.n_variables)
        masks = masks_from_subsets(subsets)
        B, K = subsets.shape
        D = self.data.n_datasets
        device = self.data.device

        result = torch.empty((B, D), dtype=torch.float64, device=device)

        missing: OrderedDict[int, torch.Tensor] = OrderedDict()
        positions: dict[int, list[int]] = {}

        for pos, (mask, subset) in enumerate(zip(masks, subsets, strict=True)):
            positions.setdefault(mask, []).append(pos)
            cached = self.cache.get(mask)
            if cached is not None:
                result[pos] = cached.to(device=device, dtype=torch.float64)
            elif mask not in missing:
                missing[mask] = subset

        if missing:
            missing_masks = list(missing.keys())
            missing_subsets = torch.stack(list(missing.values())).to(device)
            U = len(missing_masks)
            entropy_values = torch.empty((U, D), dtype=torch.float64, device=device)

            for d, dataset in enumerate(self.data.datasets):
                mode = choose_count_mode(
                    order=K,
                    n_subsets=U,
                    n_samples=dataset.shape[0],
                    requested=self.count_mode,
                    dense_state_limit=self.dense_state_limit,
                    dense_memory_limit_bytes=self.dense_memory_limit_bytes,
                )

                if mode == "dense":
                    codes = encode_binary_states(dataset, missing_subsets)
                    counts = dense_counts_from_codes(codes, 1 << K)
                    values = self.estimator.entropy_from_dense(counts.values)
                else:
                    if self.data.n_variables <= 63:
                        row_codes = self._row_codes[d]
                        if row_codes is None:
                            row_codes = encode_binary_rows(dataset)
                            self._row_codes[d] = row_codes
                        codes = mask_binary_rows(row_codes, missing_masks)
                    else:
                        codes = encode_binary_states(dataset, missing_subsets)
                    counts = sparse_counts_from_codes(codes)
                    values = self.estimator.entropy_from_sparse(counts.values)

                entropy_values[:, d] = values.to(device)

            for row, mask in enumerate(missing_masks):
                value = entropy_values[row]
                self.cache.put(mask, value)
                for pos in positions[mask]:
                    result[pos] = value

        return result
