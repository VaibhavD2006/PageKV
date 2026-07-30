"""
router.py — page routers.

Phase 1: PageRouter (flat dot-product topk).
Phase 2: HierarchicalPageRouter (two-level: super-page then page).

Both expose the same select_pages(query, page_summaries) interface so
attention.py needs no changes.
"""
import math
import torch


class PageRouter:
    """Flat dot-product page router.

    select_pages signature is the stable interface for attention.py.
    HierarchicalPageRouter below satisfies the same contract.
    """

    def __init__(self, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.top_k = top_k

    def select_pages(
        self,
        query: torch.Tensor,           # [B, H, D]
        page_summaries: torch.Tensor,  # [B, H, n_pages, D]
    ) -> torch.Tensor:                 # [B, H, k]  sorted ascending
        """Return indices of the top-k pages most relevant to query."""
        n_pages = page_summaries.shape[-2]
        k = min(self.top_k, n_pages)
        scores = torch.einsum("bhd,bhpd->bhp", query, page_summaries)
        return scores.topk(k, dim=-1).indices.sort(dim=-1).values


class DynamicPageRouter:
    """Percentage-based router: top_k scales with context length.

    Reads a fixed fraction of pages per decode step so recall quality stays
    roughly constant as context grows — unlike a fixed top_k which covers
    an ever-shrinking slice of the cache at long context.

    Example: target_pct=0.01 (1%) at 1M tokens → top_k=79 → reads 10,112 tokens.
    Same config at 100K tokens → top_k=8 → reads 1,024 tokens.

    Args:
        target_pct: fraction of pages to attend per step, in (0, 1]. 0.01 = 1%.
        min_top_k: floor so very short contexts still attend at least this many pages.
    """

    def __init__(self, target_pct: float = 0.01, min_top_k: int = 1) -> None:
        if not 0 < target_pct <= 1:
            raise ValueError(f"target_pct must be in (0, 1], got {target_pct}")
        if min_top_k < 1:
            raise ValueError(f"min_top_k must be >= 1, got {min_top_k}")
        self.target_pct = target_pct
        self.min_top_k = min_top_k

    def select_pages(
        self,
        query: torch.Tensor,           # [B, H, D]
        page_summaries: torch.Tensor,  # [B, H, n_pages, D]
    ) -> torch.Tensor:                 # [B, H, k]  sorted ascending
        n_pages = page_summaries.shape[-2]
        k = min(max(self.min_top_k, math.ceil(n_pages * self.target_pct)), n_pages)
        scores = torch.einsum("bhd,bhpd->bhp", query, page_summaries)
        return scores.topk(k, dim=-1).indices.sort(dim=-1).values


class HierarchicalPageRouter:
    """Two-level router: group pages into super-pages, route top-level first.

    Layout:
      n_pages pages  →  ceil(n_pages / super_page_size) super-pages
      Step 1: dot-product query vs super-page summaries → top_k_super super-pages
      Step 2: dot-product query vs page summaries within those super-pages → top_k pages

    This is O(n_super + top_k_super * super_page_size) instead of O(n_pages),
    which scales better when n_pages is very large.

    Args:
        top_k: final number of pages to return (matches PageRouter.top_k).
        super_page_size: how many pages form one super-page.
        top_k_super: how many super-pages to expand at level-2 (default: ceil(top_k / super_page_size)).
    """

    def __init__(
        self,
        top_k: int,
        super_page_size: int = 8,
        top_k_super: int | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if super_page_size <= 0:
            raise ValueError(f"super_page_size must be positive, got {super_page_size}")
        self.top_k = top_k
        self.super_page_size = super_page_size
        self.top_k_super = top_k_super or math.ceil(top_k / super_page_size)

    def select_pages(
        self,
        query: torch.Tensor,           # [B, H, D]
        page_summaries: torch.Tensor,  # [B, H, n_pages, D]
    ) -> torch.Tensor:                 # [B, H, top_k] sorted ascending
        """Two-level routing: super-page then page."""
        B, H, n_pages, D = page_summaries.shape

        # ── level 1: build super-page summaries (mean of page summaries) ──────
        sps = self.super_page_size
        n_super = math.ceil(n_pages / sps)
        pad = n_super * sps - n_pages
        if pad > 0:
            padding = torch.zeros(B, H, pad, D, dtype=page_summaries.dtype,
                                  device=page_summaries.device)
            ps_padded = torch.cat([page_summaries, padding], dim=2)
        else:
            ps_padded = page_summaries
        # [B, H, n_super, sps, D] -> mean -> [B, H, n_super, D]
        super_summaries = ps_padded.reshape(B, H, n_super, sps, D).mean(dim=-2)

        # ── level 2: score super-pages ────────────────────────────────────────
        k_super = min(self.top_k_super, n_super)
        super_scores = torch.einsum("bhd,bhsd->bhs", query, super_summaries)
        top_super_idx = super_scores.topk(k_super, dim=-1).indices  # [B, H, k_super]

        # ── level 3: expand super-pages to constituent page indices ───────────
        # each super-page contains sps consecutive page indices
        offsets = torch.arange(sps, device=query.device)  # [sps]
        candidate_pages = (top_super_idx.unsqueeze(-1) * sps + offsets)  # [B, H, k_super, sps]
        candidate_pages = candidate_pages.reshape(B, H, k_super * sps).clamp(0, n_pages - 1)

        # ── level 4: score candidates ─────────────────────────────────────────
        # gather summaries for candidate pages
        idx = candidate_pages.unsqueeze(-1).expand(B, H, k_super * sps, D)
        candidate_summaries = page_summaries.gather(2, idx)  # [B, H, k_super*sps, D]
        cand_scores = torch.einsum("bhd,bhcd->bhc", query, candidate_summaries)

        k_final = min(self.top_k, k_super * sps, n_pages)
        top_cand_local = cand_scores.topk(k_final, dim=-1).indices  # [B, H, top_k]
        # map local candidate index back to global page index
        top_pages = candidate_pages.gather(2, top_cand_local)  # [B, H, top_k]
        return top_pages.sort(dim=-1).values
