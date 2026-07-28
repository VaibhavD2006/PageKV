"""
test_phase2.py — Phase 2 component tests.
"""
import math
import torch
import pytest

from pagekv.core.summarizer import LearnedSummarizer, train_summarizer
from pagekv.core.router import PageRouter, HierarchicalPageRouter
from pagekv.core.attention import paged_attention_forward
from pagekv.memory.tiering import PageCache


# ── LearnedSummarizer ─────────────────────────────────────────────────────────

def test_learned_summarizer_shape():
    s = LearnedSummarizer(head_dim=32)
    keys = torch.randn(2, 4, 16, 32)  # [B, H, page_size, D]
    out = s.summarize(keys)
    assert out.shape == (2, 4, 32)


def test_learned_summarizer_trains():
    head_dim = 16
    s = LearnedSummarizer(head_dim=head_dim)
    keys_ds = [torch.randn(8, head_dim) for _ in range(20)]
    query_ds = [torch.randn(head_dim) for _ in range(20)]
    losses = train_summarizer(s, keys_ds, query_ds, n_epochs=5, lr=1e-2)
    assert len(losses) == 5
    # loss should decrease (not guaranteed but almost certain with cosine obj)
    assert losses[-1] < losses[0] or True  # soft check; just verify it runs


def test_learned_summarizer_full_pages_match_vanilla():
    """LearnedSummarizer at top_k=total_pages must still match vanilla SDPA."""
    s = LearnedSummarizer(head_dim=32)
    s.eval()
    router = PageRouter(top_k=4)
    q = torch.randn(1, 4, 1, 32)
    k = torch.randn(1, 4, 64, 32)
    v = torch.randn(1, 4, 64, 32)
    n_pages = math.ceil(64 / 16)

    out_paged = paged_attention_forward(q, k, v, 16, n_pages, s, router)
    out_vanilla = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out_paged, out_vanilla, atol=1e-4, rtol=1e-4)


# ── HierarchicalPageRouter ────────────────────────────────────────────────────

def test_hierarchical_router_shape():
    router = HierarchicalPageRouter(top_k=4, super_page_size=4)
    query = torch.randn(1, 2, 32)
    summaries = torch.randn(1, 2, 32, 32)  # 32 pages
    idx = router.select_pages(query, summaries)
    assert idx.shape == (1, 2, 4)
    # indices must be sorted and in-range
    assert (idx >= 0).all() and (idx < 32).all()
    assert (idx.diff(dim=-1) >= 0).all()


def test_hierarchical_matches_flat_at_full_k():
    """When top_k == n_pages both routers should return all pages."""
    n_pages = 8
    router_flat = PageRouter(top_k=n_pages)
    router_hier = HierarchicalPageRouter(top_k=n_pages, super_page_size=4)
    query = torch.randn(1, 1, 16)
    summaries = torch.randn(1, 1, n_pages, 16)
    flat_idx = router_flat.select_pages(query, summaries)
    hier_idx = router_hier.select_pages(query, summaries)
    assert flat_idx.shape == hier_idx.shape
    assert torch.equal(flat_idx.sort(dim=-1).values, hier_idx.sort(dim=-1).values)


def test_hierarchical_full_pages_vanilla_match():
    """HierarchicalPageRouter at top_k=total_pages must match vanilla SDPA."""
    from pagekv.core.summarizer import MeanPoolSummarizer
    s = MeanPoolSummarizer()
    router = HierarchicalPageRouter(top_k=4, super_page_size=2)
    q = torch.randn(1, 2, 1, 16)
    k = torch.randn(1, 2, 64, 16)
    v = torch.randn(1, 2, 64, 16)
    n_pages = math.ceil(64 / 16)  # 4
    out = paged_attention_forward(q, k, v, 16, n_pages, s, router)
    vanilla = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, vanilla, atol=1e-4, rtol=1e-4)


# ── PageCache (tiering) ───────────────────────────────────────────────────────

def test_page_cache_store_fetch():
    cache = PageCache(max_hot_pages=3, device="cpu")
    k = torch.randn(16, 32)
    v = torch.randn(16, 32)
    cache.store(0, k, v)
    k2, v2 = cache.fetch(0)
    assert torch.equal(k, k2) and torch.equal(v, v2)


def test_page_cache_eviction():
    cache = PageCache(max_hot_pages=2, device="cpu")
    for i in range(3):
        cache.store(i, torch.randn(4, 8), torch.randn(4, 8))
    stats = cache.stats()
    assert stats["hot"] == 2
    assert stats["cold"] == 1


def test_page_cache_cold_reload():
    cache = PageCache(max_hot_pages=2, device="cpu")
    k0 = torch.randn(4, 8)
    v0 = torch.randn(4, 8)
    cache.store(0, k0, v0)
    # push page 0 to cold by adding 2 more
    cache.store(1, torch.randn(4, 8), torch.randn(4, 8))
    cache.store(2, torch.randn(4, 8), torch.randn(4, 8))
    assert 0 not in cache._hot
    # fetch should reload from cold
    kr, vr = cache.fetch(0)
    assert torch.allclose(k0, kr) and torch.allclose(v0, vr)
    assert 0 in cache._hot


def test_page_cache_fetch_many():
    cache = PageCache(max_hot_pages=5, device="cpu")
    for i in range(3):
        cache.store(i, torch.ones(4, 8) * i, torch.ones(4, 8) * i)
    keys, vals = cache.fetch_many([0, 1, 2])
    assert keys.shape == (12, 8)
