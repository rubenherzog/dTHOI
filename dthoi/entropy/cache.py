from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class EntropyCache:
    """Small cache keyed by canonical subset masks.

    Cached tensors are stored on CPU by default so the cache does not silently
    retain accelerator memory. Values are one entropy per dataset.
    """

    def __init__(self, max_entries: int | None = None, *, store_on_cpu: bool = True):
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive or None.")
        self.max_entries = max_entries
        self.store_on_cpu = store_on_cpu
        self._data: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: int) -> torch.Tensor | None:
        value = self._data.get(key)
        if value is None:
            self.stats.misses += 1
            return None
        self._data.move_to_end(key)
        self.stats.hits += 1
        return value

    def put(self, key: int, value: torch.Tensor) -> None:
        if self.store_on_cpu:
            value = value.detach().cpu()
        else:
            value = value.detach()
        self._data[key] = value
        self._data.move_to_end(key)

        if self.max_entries is not None:
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
