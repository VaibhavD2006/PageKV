"""
attn_microbench.py -- vanilla SDPA vs paged attention (uncached / cached / compiled).

All paths are fully dynamic -- no hardcoded context lengths or model assumptions.
page_size, top_k, heads, head_dim are all CLI arguments.
"""
import argparse
import time
import math

import torch
import torch.nn.functional as F

from pagekv.core.paging import paginate_tensor
from pagekv.core.attention import _build_page_summaries
from pagekv.core.summarizer import MeanPoolSummarizer
from pagekv.core.router import PageRouter
from pagekv.integrations.hf_patch import _paged_decode, _paged_decode_compiled

HAS_COMPILE = _paged_decode_compiled is not _paged_decode


def bench(fn, warmup=5, runs=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) * 1000 / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--top-k",    type=int, default=8)
    parser.add_argument("--heads",    type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--batch",    type=int, default=1)
    args = parser.parse_args()

    B, H, D = args.batch, args.heads, args.head_dim
    ps, k   = args.page_size, args.top_k

    summarizer = MeanPoolSummarizer()
    router     = PageRouter(top_k=k)

    ctx_lens = [256, 512, 768, 1024, 1536, 2048, 4096]

    compile_label = "Compiled" if HAS_COMPILE else "Compiled(N/A)"
    if not HAS_COMPILE:
        print("  Note: torch.compile unavailable (no C++ compiler) -- showing cached path only.")

    print(f"\nAttention microbench  page_size={ps}  top_k={k}  heads={H}  head_dim={D}")
    print(f"{'Ctx':>6} {'Vanilla':>10} {'Uncached':>10} {'Cached':>10} {compile_label:>13} {'Speedup':>8}  %ctx")
    print("-" * 72)

    crossover_cached   = None
    crossover_compiled = None

    for T in ctx_lens:
        Q = torch.randn(B, H, 1, D)
        K = torch.randn(B, H, T, D)
        V = torch.randn(B, H, T, D)

        n_pages = math.ceil(T / ps)
        k_used  = min(k, n_pages)
        pct     = round(100 * k_used * ps / T)

        key_pages, _ = paginate_tensor(K, ps)
        cached_summaries = _build_page_summaries(key_pages, summarizer)

        def vanilla():
            return F.scaled_dot_product_attention(Q, K, V, is_causal=False)

        def uncached():
            kp, _ = paginate_tensor(K, ps)
            s = _build_page_summaries(kp, summarizer)
            return _paged_decode(Q, K, V, s, ps, router)

        def cached():
            return _paged_decode(Q, K, V, cached_summaries, ps, router)

        def compiled():
            return _paged_decode_compiled(Q, K, V, cached_summaries, ps, router)

        if k_used >= n_pages:
            v_ms = bench(vanilla)
            u_ms = c_ms = cp_ms = v_ms
            speedup = 1.0
            note = " (fast-path)"
        else:
            v_ms  = bench(vanilla)
            u_ms  = bench(uncached)
            c_ms  = bench(cached)
            cp_ms = bench(compiled)
            best  = cp_ms if HAS_COMPILE else c_ms
            speedup = v_ms / best
            note = ""
            if c_ms  < v_ms and crossover_cached   is None:
                crossover_cached = T
            if HAS_COMPILE and cp_ms < v_ms and crossover_compiled is None:
                crossover_compiled = T

        m = ">" if speedup > 1.0 else " "
        print(f"{T:>6,} {v_ms:>9.3f}ms {u_ms:>9.3f}ms {c_ms:>9.3f}ms {cp_ms:>12.3f}ms  {m}{speedup:>5.2f}x  {pct}%{note}")

    print()
    if crossover_cached:
        print(f"  Cached crossover:    ~{crossover_cached} tokens")
    if crossover_compiled:
        print(f"  Compiled crossover:  ~{crossover_compiled} tokens")
    elif HAS_COMPILE:
        print("  Compiled crossover:  beyond tested range")
    print(f"  KV attended per step (constant regardless of T): {k*ps} tokens")
    print(f"  Tune --top-k and --page-size for your quality/speed tradeoff.")

if __name__ == "__main__":
    main()
