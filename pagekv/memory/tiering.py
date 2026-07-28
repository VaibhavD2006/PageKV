"""
tiering.py — LRU-based GPU-hot / CPU-cold KV page cache.

Pages are evicted from GPU to CPU RAM when the hot tier exceeds max_hot_pages.
On router selection, cold pages are reloaded to GPU on demand.
"""
from __future__ import annotations
from collections import OrderedDict

import torch


class PageCache:
    """LRU two-tier KV page cache.

    Hot tier: GPU (or whatever device the model runs on).
    Cold tier: CPU pinned memory.

    Args:
        max_hot_pages: maximum pages kept on GPU at once.
        device: GPU device string, e.g. "cuda:0".
    """

    def __init__(self, max_hot_pages: int, device: str = "cuda") -> None:
        if max_hot_pages <= 0:
            raise ValueError(f"max_hot_pages must be positive, got {max_hot_pages}")
        self.max_hot_pages = max_hot_pages
        self.device = device
        # OrderedDict used as LRU: most-recently-used at end
        self._hot: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cold: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # ── write ─────────────────────────────────────────────────────────────────

    def store(self, page_id: int, key_page: torch.Tensor, val_page: torch.Tensor) -> None:
        """Store a K/V page. Immediately placed in hot tier; evicts LRU if full.

        Args:
            page_id: integer identifier for this page.
            key_page: [page_size, D] key tensor on GPU.
            val_page: [page_size, D] value tensor on GPU.
        """
        if page_id in self._hot:
            self._hot.move_to_end(page_id)
            self._hot[page_id] = (key_page, val_page)
            return

        if len(self._hot) >= self.max_hot_pages:
            self._evict_lru()

        self._hot[page_id] = (key_page, val_page)

    # ── read ──────────────────────────────────────────────────────────────────

    def fetch(self, page_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (key_page, val_page) on GPU, promoting from cold if needed."""
        if page_id in self._hot:
            self._hot.move_to_end(page_id)
            return self._hot[page_id]

        if page_id in self._cold:
            k, v = self._cold.pop(page_id)
            k = k.to(self.device, non_blocking=True)
            v = v.to(self.device, non_blocking=True)
            self.store(page_id, k, v)
            return self._hot[page_id]

        raise KeyError(f"Page {page_id} not found in hot or cold tier")

    def fetch_many(self, page_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch multiple pages and concatenate along the sequence dimension.

        Returns:
            keys: [len(page_ids) * page_size, D]
            vals: [len(page_ids) * page_size, D]
        """
        ks, vs = zip(*(self.fetch(pid) for pid in page_ids))
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    # ── eviction ──────────────────────────────────────────────────────────────

    def _evict_lru(self) -> None:
        """Move least-recently-used hot page to CPU cold tier."""
        page_id, (k, v) = self._hot.popitem(last=False)
        # pin_memory() makes H2D transfers faster on next reload
        kc, vc = k.cpu(), v.cpu()
        if torch.cuda.is_available():
            kc, vc = kc.pin_memory(), vc.pin_memory()
        self._cold[page_id] = (kc, vc)

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return {"hot": len(self._hot), "cold": len(self._cold)}

    def clear(self) -> None:
        self._hot.clear()
        self._cold.clear()
