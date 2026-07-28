"""
test_correctness.py — verify paged attention matches vanilla at top_k=total_pages.

Assumption: uses gpt2 (124M) on CPU; no GPU required.
Run: pytest pagekv/tests/test_correctness.py -v
"""
import copy
import math
import pytest
import torch

from pagekv.core.paging import paginate_tensor, get_page_token_indices
from pagekv.core.summarizer import MeanPoolSummarizer, MaxPoolSummarizer
from pagekv.core.router import PageRouter
from pagekv.core.attention import paged_attention_forward


# ── helpers ───────────────────────────────────────────────────────────────────

ATOL = 1e-4
RTOL = 1e-4


def vanilla_sdpa(q, k, v):
    Sq = q.shape[-2]
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=(Sq > 1)
    )


def make_qkv(B=1, H=4, Sq=1, Skv=64, D=32, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    q = torch.randn(B, H, Sq, D, generator=g)
    k = torch.randn(B, H, Skv, D, generator=g)
    v = torch.randn(B, H, Skv, D, generator=g)
    return q, k, v


# ── paging unit tests ─────────────────────────────────────────────────────────

def test_paginate_no_padding():
    x = torch.ones(1, 2, 16, 8)
    pages, real_len = paginate_tensor(x, 8)
    assert pages.shape == (1, 2, 2, 8, 8)
    assert real_len == 16


def test_paginate_with_padding():
    x = torch.ones(1, 2, 10, 8)
    pages, real_len = paginate_tensor(x, 8)
    assert pages.shape == (1, 2, 2, 8, 8)
    assert real_len == 10
    # padded tokens should be zero
    assert pages[:, :, -1, -6:, :].sum().item() == 0.0


# ── fast-path correctness: paged == vanilla when top_k >= n_pages ─────────────

@pytest.mark.parametrize("page_size", [8, 16, 32])
def test_full_pages_match_vanilla(page_size):
    q, k, v = make_qkv(B=1, H=4, Sq=1, Skv=64)
    n_pages = math.ceil(64 / page_size)
    summarizer = MeanPoolSummarizer()
    router = PageRouter(top_k=n_pages)

    out_paged = paged_attention_forward(q, k, v, page_size, n_pages, summarizer, router)
    out_vanilla = vanilla_sdpa(q, k, v)

    assert torch.allclose(out_paged, out_vanilla, atol=ATOL, rtol=RTOL), (
        f"Max diff: {(out_paged - out_vanilla).abs().max().item():.2e}"
    )


def test_short_sequence_fallback():
    """seq_len <= page_size triggers fast-path, output matches vanilla."""
    q, k, v = make_qkv(B=1, H=2, Sq=1, Skv=8)
    summarizer = MeanPoolSummarizer()
    router = PageRouter(top_k=4)

    out_paged = paged_attention_forward(q, k, v, page_size=16, top_k_pages=4, summarizer=summarizer, router=router)
    out_vanilla = vanilla_sdpa(q, k, v)
    assert torch.allclose(out_paged, out_vanilla, atol=ATOL, rtol=RTOL)


def test_max_summarizer_full_pages_match_vanilla():
    q, k, v = make_qkv(B=1, H=4, Sq=1, Skv=64)
    n_pages = math.ceil(64 / 16)
    summarizer = MaxPoolSummarizer()
    router = PageRouter(top_k=n_pages)

    out_paged = paged_attention_forward(q, k, v, 16, n_pages, summarizer, router)
    out_vanilla = vanilla_sdpa(q, k, v)
    assert torch.allclose(out_paged, out_vanilla, atol=ATOL, rtol=RTOL)


def test_batch_size_2():
    q, k, v = make_qkv(B=2, H=4, Sq=1, Skv=64)
    n_pages = math.ceil(64 / 16)
    summarizer = MeanPoolSummarizer()
    router = PageRouter(top_k=n_pages)

    out_paged = paged_attention_forward(q, k, v, 16, n_pages, summarizer, router)
    out_vanilla = vanilla_sdpa(q, k, v)
    assert torch.allclose(out_paged, out_vanilla, atol=ATOL, rtol=RTOL)


# ── HuggingFace model-level correctness ───────────────────────────────────────

def _load_gpt2():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return model, tokenizer
    except Exception:
        pytest.skip("gpt2 not available (network or transformers missing)")


def test_hf_patch_matches_vanilla_logits():
    """Patched model at top_k=total_pages should produce same logits as unpatched."""
    from pagekv import patch_model

    model_vanilla, tokenizer = _load_gpt2()
    model_paged = copy.deepcopy(model_vanilla)

    text = "The quick brown fox jumps over the lazy dog"
    ids = tokenizer(text, return_tensors="pt").input_ids  # seq_len ~ 10

    # Use a page_size larger than seq_len so fast-path (original _attn) fires
    patch_model(model_paged, page_size=128, top_k_pages=100)

    model_vanilla.eval()
    model_paged.eval()
    with torch.no_grad():
        logits_v = model_vanilla(ids).logits
        logits_p = model_paged(ids).logits

    assert torch.allclose(logits_v, logits_p, atol=1e-3, rtol=1e-3), (
        f"Max logit diff: {(logits_v - logits_p).abs().max().item():.2e}"
    )


def test_hf_patch_generate_runs():
    """Patched model.generate() completes without error."""
    from pagekv import patch_model

    model, tokenizer = _load_gpt2()
    tokenizer.pad_token = tokenizer.eos_token
    patch_model(model, page_size=4, top_k_pages=2)
    model.eval()

    ids = tokenizer("Hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=5, do_sample=False)
    assert out.shape[-1] > ids.shape[-1]
