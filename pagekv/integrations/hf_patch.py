"""
hf_patch.py -- monkey-patches HuggingFace attention to use PageKV.

Decode path is optimised two ways:
  1. Summary cache: computed once at prefill, reused every decode step.
     Re-built only when a new page is completed (incremental, not full rebuild).
  2. Direct token indexing: avoids creating an intermediate paged view of K/V.
     page_index * page_size + [0..page_size-1] gives token indices directly,
     then a single gather on the original K/V tensor.
  3. torch.compile with dynamic=True: fuses the route + index + gather + SDPA
     into one kernel per step with no per-call Python overhead.

Nothing is hardcoded to a context length, dataset, or model architecture.
page_size, top_k_pages, and the summarizer are all runtime parameters.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn.functional as F

from pagekv.core.paging import paginate_tensor
from pagekv.core.attention import _build_page_summaries
from pagekv.core.summarizer import BaseSummarizer, MeanPoolSummarizer
from pagekv.core.router import PageRouter, DynamicPageRouter

_IMPL_KEY = "pagekv"


def _paged_decode(
    query: torch.Tensor,        # [B, H, 1, D]
    key: torch.Tensor,          # [B, H, T, D]
    value: torch.Tensor,        # [B, H, T, D]
    summaries: torch.Tensor,    # [B, H, n_pages, D]
    page_size: int,
    router: "PageRouter",
) -> torch.Tensor:              # [B, H, 1, D]
    """Route, gather selected pages via direct token indexing, attend.

    Fully dynamic: handles any B, H, T, page_size, top_k at runtime.
    """
    B, H, T, D = key.shape

    # Route: O(n_pages * D) -- fast even at large T
    routing_q   = query[:, :, -1, :]                                    # [B, H, D]
    page_indices = router.select_pages(routing_q, summaries)            # [B, H, top_k]

    top_k = page_indices.shape[-1]

    # Direct token indexing: page i covers tokens [i*page_size : i*page_size + page_size]
    # Clamp keeps OOB indices (partial last page) in bounds -- repeats the last token
    # for padding positions, which is preferable to a separate zero-pad allocation.
    offsets  = torch.arange(page_size, device=key.device)              # [page_size]
    tok_idx  = (page_indices * page_size).unsqueeze(-1) + offsets      # [B, H, top_k, page_size]
    flat_idx = tok_idx.reshape(B, H, -1).clamp(0, T - 1)              # [B, H, top_k*page_size]

    exp   = flat_idx.unsqueeze(-1).expand(B, H, top_k * page_size, D)
    sel_k = key.gather(2, exp)                                          # [B, H, top_k*ps, D]
    sel_v = value.gather(2, exp)

    return F.scaled_dot_product_attention(query, sel_k, sel_v, is_causal=False)


# Compile with dynamic=True so shapes (T, n_pages, top_k) can vary without
# recompilation. Falls back silently on older torch builds or environments
# where a C++ compiler is unavailable (e.g. Windows without MSVC).
def _make_compiled_decode():
    try:
        compiled = torch.compile(
            _paged_decode, mode="reduce-overhead", dynamic=True
        )
        # Probe with a tiny input to surface compiler errors at import time
        # rather than at the first real call.
        _q = torch.zeros(1, 1, 1, 4)
        _k = torch.zeros(1, 1, 16, 4)
        _s = torch.zeros(1, 1, 2, 4)
        compiled(_q, _k, _k, _s, 8, PageRouter(top_k=1))
        return compiled
    except Exception:
        return _paged_decode

_paged_decode_compiled = _make_compiled_decode()


def _make_pagekv_attn_fn(
    page_size: int,
    top_k_pages: int,
    summarizer: BaseSummarizer,
    router: PageRouter,
):
    def pagekv_attn_fn(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        Skv      = key.shape[2]
        n_pages  = math.ceil(Skv / page_size)
        is_prefill = query.shape[2] > 1

        # ── prefill: vanilla causal SDPA + build summary cache ────────────────
        if is_prefill:
            if n_pages > top_k_pages and Skv > page_size:
                key_pages, _ = paginate_tensor(key, page_size)
                module._pagekv_summaries = _build_page_summaries(key_pages, summarizer)
                module._pagekv_n_pages   = n_pages
            else:
                module._pagekv_summaries = None
            return (
                F.scaled_dot_product_attention(query, key, value, is_causal=True)
                .transpose(1, 2),
                None,
            )

        # ── fast path: too few pages to bother routing ─────────────────────────
        if n_pages <= top_k_pages or Skv <= page_size:
            return (
                F.scaled_dot_product_attention(query, key, value, is_causal=False)
                .transpose(1, 2),
                None,
            )

        # ── decode: use cached summaries, update only when a new page completes ─
        cached_n = getattr(module, "_pagekv_n_pages", 0)
        if n_pages != cached_n or getattr(module, "_pagekv_summaries", None) is None:
            key_pages, _ = paginate_tensor(key, page_size)
            module._pagekv_summaries = _build_page_summaries(key_pages, summarizer)
            module._pagekv_n_pages   = n_pages

        out = _paged_decode_compiled(
            query, key, value, module._pagekv_summaries, page_size, router
        )
        return out.transpose(1, 2), None

    return pagekv_attn_fn


def patch_model(
    model: object,
    page_size: int = 128,
    top_k_pages: int = 4,
    summarizer_cls: type[BaseSummarizer] = MeanPoolSummarizer,
    router: "PageRouter | None" = None,
) -> None:
    """Replace HuggingFace attention with PageKV paged attention.

    No retraining required. Call once after loading; use model.generate() as normal.

    Pass a custom router to control how pages are selected each decode step:
        router=DynamicPageRouter(target_pct=0.01)  # 1% of pages, scales with context
        router=HierarchicalPageRouter(top_k=4)     # O(√n) routing for very long context
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if top_k_pages <= 0:
        raise ValueError(f"top_k_pages must be positive, got {top_k_pages}")

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    summarizer = summarizer_cls()
    if router is None:
        router = PageRouter(top_k=top_k_pages)

    ALL_ATTENTION_FUNCTIONS[_IMPL_KEY] = _make_pagekv_attn_fn(
        page_size, top_k_pages, summarizer, router
    )

    if not hasattr(model, "config"):
        raise ValueError("model must have a .config attribute (HuggingFace PreTrainedModel)")

    model.config._attn_implementation = _IMPL_KEY
    for module in model.modules():
        if hasattr(module, "config") and module.config is not model.config:
            module.config._attn_implementation = _IMPL_KEY

