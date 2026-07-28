"""
attention.py — drop-in paged attention forward pass.

When top_k_pages >= n_pages (or seq is shorter than one page) this is
numerically identical to vanilla scaled_dot_product_attention — that property
is what test_correctness.py verifies.
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F

from pagekv.core.paging import paginate_tensor
from pagekv.core.summarizer import BaseSummarizer
from pagekv.core.router import PageRouter


def _build_page_summaries(
    key_pages: torch.Tensor,      # [B, H, n_pages, page_size, D]
    summarizer: BaseSummarizer,
) -> torch.Tensor:                # [B, H, n_pages, D]
    """Vectorised summary: reshape so summarizer sees [B, H, page_size, D] per page."""
    B, H, n_pages, page_size, D = key_pages.shape
    # treat (B * n_pages) as the batch dimension
    batched = key_pages.permute(0, 2, 1, 3, 4).reshape(B * n_pages, H, page_size, D)
    flat = summarizer.summarize(batched)                  # [B*n_pages, H, D]
    return flat.reshape(B, n_pages, H, D).permute(0, 2, 1, 3)  # [B, H, n_pages, D]


def paged_attention_forward(
    query: torch.Tensor,      # [B, H, Sq, D]
    key: torch.Tensor,        # [B, H, Skv, D]
    value: torch.Tensor,      # [B, H, Skv, D]
    page_size: int,
    top_k_pages: int,
    summarizer: BaseSummarizer,
    router: PageRouter,
    scale: Optional[float] = None,
) -> torch.Tensor:            # [B, H, Sq, D]
    """Paged attention: route to top-k pages, run full attention only there.

    Fast-path (top_k_pages >= n_pages OR Skv <= page_size):
        Equivalent to F.scaled_dot_product_attention — no routing overhead.

    Paged-path:
        1. Paginate K/V into [B, H, n_pages, page_size, D].
        2. Summarize each page.
        3. Route using the last query token (decode hot-path).
        4. Gather selected pages, run SDPA on the reduced K/V.
    """
    B, H, Skv, D = key.shape
    n_pages = math.ceil(Skv / page_size)
    is_causal = query.shape[-2] > 1  # causal mask needed for prefill only

    # ── fast path ─────────────────────────────────────────────────────────────
    if top_k_pages >= n_pages or Skv <= page_size:
        return F.scaled_dot_product_attention(
            query, key, value,
            scale=scale,
            is_causal=is_causal,
        )

    # ── paged path ────────────────────────────────────────────────────────────
    key_pages, real_kv_len = paginate_tensor(key, page_size)    # [B, H, n_pages, ps, D]
    value_pages, _ = paginate_tensor(value, page_size)

    page_summaries = _build_page_summaries(key_pages, summarizer)  # [B, H, n_pages, D]

    routing_query = query[:, :, -1, :]  # [B, H, D] — last token routes
    page_indices = router.select_pages(routing_query, page_summaries)  # [B, H, top_k]

    top_k = page_indices.shape[-1]
    # gather: expand indices to [B, H, top_k, page_size, D]
    idx = page_indices.unsqueeze(-1).unsqueeze(-1).expand(B, H, top_k, page_size, D)
    selected_key = key_pages.gather(2, idx).reshape(B, H, top_k * page_size, D)
    selected_val = value_pages.gather(2, idx).reshape(B, H, top_k * page_size, D)

    # paged path is always non-causal: we've already selected relevant (past) pages
    return F.scaled_dot_product_attention(
        query, selected_key, selected_val,
        scale=scale,
        is_causal=False,
    )
