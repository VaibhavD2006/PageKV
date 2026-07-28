"""
hf_patch.py — monkey-patches HuggingFace attention to use PageKV.

Phase 1 approach: registers a "pagekv" implementation in transformers'
ALL_ATTENTION_FUNCTIONS registry, then sets model.config._attn_implementation
so every attention layer routes through the paged forward.
"""
from __future__ import annotations
import math
from typing import Optional

import torch

from pagekv.core.attention import paged_attention_forward
from pagekv.core.summarizer import BaseSummarizer, MeanPoolSummarizer
from pagekv.core.router import PageRouter


# ── attention interface (matches HF signature) ─────────────────────────────────

def _make_pagekv_attn_fn(
    page_size: int,
    top_k_pages: int,
    summarizer: BaseSummarizer,
    router: PageRouter,
):
    """Return an attention function compatible with ALL_ATTENTION_FUNCTIONS."""

    def pagekv_attn_fn(
        module,
        query: torch.Tensor,           # [B, H, Sq, D]
        key: torch.Tensor,             # [B, H, Skv, D]
        value: torch.Tensor,           # [B, H, Skv, D]
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        out = paged_attention_forward(
            query, key, value,
            page_size=page_size,
            top_k_pages=top_k_pages,
            summarizer=summarizer,
            router=router,
        )
        # HF expects (attn_output transposed, attn_weights)
        # paged_attention_forward returns [B, H, Sq, D]; HF forward() transposes after
        # so we return it as-is and let the caller handle the rest.
        # NOTE: eager_attention_forward does .transpose(1,2) at the end;
        # our paged_attention_forward returns [B, H, Sq, D] so we match that pre-transpose shape.
        return out.transpose(1, 2), None

    return pagekv_attn_fn


# ── public API ────────────────────────────────────────────────────────────────

_IMPL_KEY = "pagekv"


def patch_model(
    model: object,
    page_size: int = 128,
    top_k_pages: int = 4,
    summarizer_cls: type[BaseSummarizer] = MeanPoolSummarizer,
) -> None:
    """Replace HuggingFace attention with PageKV paged attention.

    No retraining required. Call once after loading the model; then use
    model.generate() / model() as normal.

    Phase 1: GPT-2 architecture via transformers' ALL_ATTENTION_FUNCTIONS.
    Add support for other architectures by ensuring their configs expose
    _attn_implementation and they use ALL_ATTENTION_FUNCTIONS dispatch.
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if top_k_pages <= 0:
        raise ValueError(f"top_k_pages must be positive, got {top_k_pages}")

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    summarizer = summarizer_cls()
    router = PageRouter(top_k=top_k_pages)

    ALL_ATTENTION_FUNCTIONS[_IMPL_KEY] = _make_pagekv_attn_fn(
        page_size, top_k_pages, summarizer, router
    )

    # Set on the top-level config; GPT-2 layers read from model.config
    if not hasattr(model, "config"):
        raise ValueError("model must have a .config attribute (HuggingFace PreTrainedModel)")

    model.config._attn_implementation = _IMPL_KEY

    # Also update sub-module configs where they exist (some models propagate per-layer)
    for module in model.modules():
        if hasattr(module, "config") and module.config is not model.config:
            module.config._attn_implementation = _IMPL_KEY
