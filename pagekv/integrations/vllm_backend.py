"""
vllm_backend.py — PageKV custom attention backend for vLLM.

Implements vLLM's AttentionBackend / AttentionImpl plugin interface.
Requires: pip install vllm>=0.4.0

Registration:
    Set the environment variable before starting vLLM:
        VLLM_ATTENTION_BACKEND=pagekv

    Or pass to the engine:
        LLM(model="...", attention_backend="pagekv")

Design notes:
    vLLM manages its own paged KV cache (physical blocks). This backend
    receives pre-fetched K/V tensors for the selected slots and applies
    PageKV's summarise-and-route step on top, further reducing how many
    vLLM KV blocks are loaded into the attention window per step.

    Phase 3 implementation targets vLLM >= 0.4.0 which exposes:
        AttentionBackend  (class, registered via entry_points)
        AttentionImpl     (per-layer instance, called in model forward)
        AttentionMetadata (batched scheduling metadata)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch

from pagekv.core.attention import paged_attention_forward
from pagekv.core.summarizer import BaseSummarizer, MeanPoolSummarizer
from pagekv.core.router import PageRouter

if TYPE_CHECKING:
    # vLLM types; imported lazily so pagekv works without vLLM installed
    from vllm.attention.backends.abstract import AttentionMetadata  # noqa: F401


# ── metadata ──────────────────────────────────────────────────────────────────

@dataclass
class PageKVMetadata:
    """Per-forward-pass routing metadata (passed through vLLM's attention_metadata)."""
    page_size: int
    top_k_pages: int
    is_prefill: bool


# ── backend (registered with vLLM's plugin system) ────────────────────────────

class PageKVAttentionBackend:
    """vLLM AttentionBackend implementation for PageKV.

    vLLM discovers backends via Python entry_points under the group
    'vllm.attention_backends'. The pyproject.toml entry_point for this class:

        [project.entry-points."vllm.attention_backends"]
        pagekv = "pagekv.integrations.vllm_backend:PageKVAttentionBackend"
    """

    @staticmethod
    def get_name() -> str:
        return "pagekv"

    @staticmethod
    def get_impl_cls():
        return PageKVAttentionImpl

    @staticmethod
    def get_metadata_cls():
        return PageKVMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        # Match vLLM's default paged layout: [2, num_blocks, block_size, num_kv_heads, head_size]
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(src: torch.Tensor, dst: torch.Tensor, block_mapping: dict) -> None:
        """Copy KV blocks between GPU and CPU (tiering hook)."""
        for src_idx, dst_idx in block_mapping.items():
            dst[src_idx] = src[dst_idx]

    @staticmethod
    def copy_blocks(kv_caches: list[torch.Tensor], block_mapping: dict) -> None:
        for src_idx, dst_indices in block_mapping.items():
            for dst_idx in dst_indices:
                for kv in kv_caches:
                    kv[dst_idx] = kv[src_idx]


# ── per-layer implementation ───────────────────────────────────────────────────

class PageKVAttentionImpl:
    """Per-layer attention implementation called by vLLM's model executor.

    vLLM calls:
        impl = PageKVAttentionImpl(num_heads, head_size, scale, ...)
        output = impl.forward(query, key, value, kv_cache, attn_metadata)
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        # PageKV-specific
        page_size: int = 128,
        top_k_pages: int = 4,
        summarizer: Optional[BaseSummarizer] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.page_size = page_size
        self.top_k_pages = top_k_pages
        self.summarizer = summarizer or MeanPoolSummarizer()
        self.router = PageRouter(top_k=top_k_pages)

    def forward(
        self,
        query: torch.Tensor,          # [total_tokens, num_heads * head_size]
        key: torch.Tensor,            # [total_tokens, num_kv_heads * head_size]
        value: torch.Tensor,          # [total_tokens, num_kv_heads * head_size]
        kv_cache: torch.Tensor,       # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: Any,
    ) -> torch.Tensor:                # [total_tokens, num_heads * head_size]
        """PageKV attention forward compatible with vLLM's execution model.

        For prefill sequences: falls back to full attention (no routing needed —
        we are building the KV cache, not reading it).

        For decode steps: applies page routing over the accumulated KV cache.
        """
        B_H = self.num_heads
        D = self.head_size

        # Reshape to multi-head: [1, H, T, D] (vLLM passes flat tokens)
        T = query.shape[0]
        q = query.reshape(1, T, B_H, D).transpose(1, 2)   # [1, H, T, D]
        k = key.reshape(1, T, self.num_kv_heads, D).transpose(1, 2)
        v = value.reshape(1, T, self.num_kv_heads, D).transpose(1, 2)

        # Determine is_prefill from metadata (vLLM sets this)
        is_prefill = getattr(attn_metadata, "is_prefill", True)

        out = paged_attention_forward(
            q, k, v,
            page_size=self.page_size,
            top_k_pages=self.top_k_pages if not is_prefill else 2**31,
            summarizer=self.summarizer,
            router=self.router,
            scale=self.scale,
        )  # [1, H, T, D]

        # Reshape back to vLLM's flat layout [T, H * D]
        return out.transpose(1, 2).reshape(T, B_H * D)
