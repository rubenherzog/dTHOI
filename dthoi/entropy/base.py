from __future__ import annotations

import torch

from ..data import DiscreteData, DiscreteInput, prepare_discrete_data
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

    This is an internal execution component. Scientific users should normally
    call :func:`dthoi.estimate_entropy` or :func:`dthoi.information_measures`.

    The provider owns the shared ``variable set -> entropy`` cache. Local
    entropy values are computed only when requested and are never stored in the
    persistent cache because their size scales with the number of samples.

    Parameters
    ----------
    X
        Discrete observations accepted by :func:`dthoi.prepare_data`.
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
        self.data: DiscreteData = prepare_discrete_data(X, device=device)
        self.estimator = resolve_estimator(estimator)
        self.count_mode = count_mode
        self.cache = cache if cache is not None else EntropyCache()
        self.dense_state_limit = dense_state_limit
        self.dense_memory_limit_bytes = dense_memory_limit_bytes
        self._row_codes: list[torch.Tensor | None] = [None] * self.data.n_datasets

    def _estimate_uncached(
        self,
        subsets: torch.Tensor,
        masks: list[int],
        *,
        return_local: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Estimate uncached entropy terms, optionally retaining sample values.

        Parameters
        ----------
        subsets
            Canonical fixed-order variable sets with shape ``[B, K]``.
        masks
            Canonical integer masks corresponding one-to-one with ``subsets``.
        return_local
            Whether to also calculate entropy values for every observation.

        Returns
        -------
        tuple
            Global entropy with shape ``[B, D]`` and, when requested, one local
            tensor per dataset with shape ``[B, T_d]``. All values are in bits.
        """
        if return_local and not self.estimator.supports_local_values:
            raise NotImplementedError(
                f"Entropy estimator {self.estimator.name!r} does not define local values."
            )

        n_subsets, order = subsets.shape
        n_datasets = self.data.n_datasets
        device = self.data.device
        entropy_values = torch.empty(
            (n_subsets, n_datasets), dtype=torch.float64, device=device
        )
        local_values: list[torch.Tensor] | None = [] if return_local else None

        for dataset_index, dataset in enumerate(self.data.datasets):
            mode = choose_count_mode(
                order=order,
                n_subsets=n_subsets,
                n_samples=dataset.shape[0],
                requested=self.count_mode,
                dense_state_limit=self.dense_state_limit,
                dense_memory_limit_bytes=self.dense_memory_limit_bytes,
            )

            if mode == "dense":
                codes = encode_binary_states(dataset, subsets)
                counts = dense_counts_from_codes(codes, 1 << order)
                values = self.estimator.entropy_from_dense(counts)
                if return_local:
                    assert local_values is not None
                    local_values.append(self.estimator.local_from_dense(counts, codes))
            else:
                if self.data.n_variables <= 63:
                    row_codes = self._row_codes[dataset_index]
                    if row_codes is None:
                        row_codes = encode_binary_rows(dataset)
                        self._row_codes[dataset_index] = row_codes
                    codes = mask_binary_rows(row_codes, masks)
                else:
                    codes = encode_binary_states(dataset, subsets)

                if return_local:
                    sparse_result = sparse_counts_from_codes(codes, return_inverse=True)
                    counts, inverse = sparse_result
                    values = self.estimator.entropy_from_sparse(counts)
                    assert local_values is not None
                    local_values.append(self.estimator.local_from_sparse(counts, inverse))
                else:
                    counts = sparse_counts_from_codes(codes)
                    assert isinstance(counts, list)
                    values = self.estimator.entropy_from_sparse(counts)

            entropy_values[:, dataset_index] = values.to(device=device, dtype=torch.float64)

        return entropy_values, None if local_values is None else tuple(local_values)

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
        batch_size = subsets.shape[0]
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
        entropy_values, _ = self._estimate_uncached(
            missing_subsets,
            missing_masks,
            return_local=False,
        )

        for row, mask in enumerate(missing_masks):
            value = entropy_values[row]
            self.cache.put(mask, value)
            for position in positions[mask]:
                result[position] = value

        return result

    def entropy_with_local(
        self,
        subsets: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Estimate global and sample-resolved entropy for variable sets.

        Local values are aligned with the original sample order of each dataset.
        For the empirical estimator they are Shannon surprisals in bits. An
        estimator may define another documented decomposition; estimators that
        do not support local values raise :class:`NotImplementedError`.

        Parameters
        ----------
        subsets
            Integer variable sets with shape ``[B, K]`` or ``[K]``.

        Returns
        -------
        tuple
            ``(entropy, local_entropy)``. ``entropy`` has shape ``[B, D]``.
            ``local_entropy`` is a tuple with one tensor per dataset, each with
            shape ``[B, T_d]``. Datasets with unequal sample counts remain
            separate; no padding or reshaping of observations is introduced.

        Notes
        -----
        Persistent caching is intentionally limited to global entropy values.
        Local arrays scale as ``variable sets × samples`` and are therefore
        kept only for the duration of the requesting batch.
        """
        subsets = canonicalize_subsets(subsets, self.data.n_variables)
        masks = _masks_from_canonical_subsets(subsets)
        device = self.data.device

        unique_masks: list[int] = []
        unique_subsets: list[torch.Tensor] = []
        unique_index: dict[int, int] = {}
        inverse_positions: list[int] = []

        for mask, subset in zip(masks, subsets, strict=True):
            index = unique_index.get(mask)
            if index is None:
                index = len(unique_masks)
                unique_index[mask] = index
                unique_masks.append(mask)
                unique_subsets.append(subset)
            inverse_positions.append(index)

        stacked_subsets = torch.stack(unique_subsets).to(device)
        entropy_unique, local_unique = self._estimate_uncached(
            stacked_subsets,
            unique_masks,
            return_local=True,
        )
        assert local_unique is not None

        for row, mask in enumerate(unique_masks):
            self.cache.put(mask, entropy_unique[row])

        if len(unique_masks) == len(masks):
            return entropy_unique, local_unique

        gather_index = torch.tensor(inverse_positions, dtype=torch.long, device=device)
        entropy = entropy_unique.index_select(0, gather_index)
        local = tuple(values.index_select(0, gather_index) for values in local_unique)
        return entropy, local
