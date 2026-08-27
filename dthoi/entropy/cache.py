from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class CacheStats:
    """Mutable counters describing entropy-cache activity."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0


class EntropyCache:
    """LRU cache of subset entropies keyed by canonical subset masks.

    Parameters
    ----------
    max_entries
        Maximum number of cached subsets. ``None`` disables eviction.
    store_on_cpu
        If ``True``, detach cached tensors and store them on CPU so cache growth
        cannot silently retain accelerator memory.
    """

    def __init__(self, max_entries: int | None = None, *, store_on_cpu: bool = True):
        """Initialize an empty entropy cache."""
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive or None.")
        self.max_entries = max_entries
        self.store_on_cpu = store_on_cpu
        self._data: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: int) -> torch.Tensor | None:
        """Return a cached entropy vector and update LRU/statistics state."""
        value = self._data.get(key)
        if value is None:
            self.stats.misses += 1
            return None
        self._data.move_to_end(key)
        self.stats.hits += 1
        return value

    def put(self, key: int, value: torch.Tensor) -> None:
        """Insert or replace one entropy vector and enforce the LRU limit."""
        value = value.detach().cpu() if self.store_on_cpu else value.detach()
        self._data[key] = value
        self._data.move_to_end(key)

        if self.max_entries is not None:
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        """Remove cached values without resetting accumulated statistics."""
        self._data.clear()

    def __len__(self) -> int:
        """Return the number of currently cached subsets."""
        return len(self._data)
