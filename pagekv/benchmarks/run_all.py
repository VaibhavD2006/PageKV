"""
run_all.py — orchestrate benchmark matrix and emit results CSV + chart.

Usage:
    python pagekv/benchmarks/run_all.py --device cpu --ctx-lens 256 512 1024 --model gpt2

Produces:
    pagekv/benchmarks/output/results.csv
    pagekv/benchmarks/output/memory_vs_context.png
"""
from __future__ import annotations
import argparse
import copy
import csv
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch


# ── result row ────────────────────────────────────────────────────────────────

@dataclass
class BenchRow:
    context_length: int
    config: str
    peak_memory_gb: float
    latency_ms_per_tok: float
    needle_accuracy: float


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    model,
    ctx_len: int,
    config_name: str,
    device: str,
    needle_trials: int,
) -> BenchRow:
    from pagekv.benchmarks.memory_profile import measure_peak_memory_gb, measure_decode_latency_ms
    from pagekv.benchmarks.needle_in_haystack import NeedleConfig, run_needle_benchmark, generate_needle_sequence

    # Build a synthetic input; cap to model max context (e.g. GPT-2 = 1024)
    max_ctx = getattr(model.config, 'max_position_embeddings', ctx_len)
    safe_len = min(ctx_len, max_ctx - 10)  # leave room for generated tokens
    input_ids = torch.randint(0, 50257, (1, safe_len), dtype=torch.long)

    peak_mem = measure_peak_memory_gb(model, input_ids, device)
    latency = measure_decode_latency_ms(model, input_ids, n_new_tokens=5, device=device)

    needle_cfg = NeedleConfig(
        ctx_len=ctx_len,
        page_size=0,          # unused in needle_in_haystack; set by caller
        top_k_pages=0,
        config_name=config_name,
        n_trials=needle_trials,
    )
    needle_acc = run_needle_benchmark(model, needle_cfg, device=device)

    return BenchRow(
        context_length=ctx_len,
        config=config_name,
        peak_memory_gb=round(peak_mem, 4),
        latency_ms_per_tok=round(latency, 2),
        needle_accuracy=round(needle_acc, 3),
    )


# ── chart ─────────────────────────────────────────────────────────────────────

def plot_results(rows: list[BenchRow], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    vanilla = {r.context_length: r.peak_memory_gb for r in rows if r.config == "vanilla"}
    pagekv  = {r.context_length: r.peak_memory_gb for r in rows if r.config.startswith("pagekv")}

    ctx_lens = sorted(set(r.context_length for r in rows))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ctx_lens, [vanilla.get(c, 0) for c in ctx_lens], marker="o", label="Vanilla")
    ax.plot(ctx_lens, [pagekv.get(c, 0) for c in ctx_lens],  marker="s", label="PageKV")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("PageKV vs Vanilla: Peak KV Cache Memory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PageKV benchmark suite")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ctx-lens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--top-k-pages", type=int, default=4)
    parser.add_argument("--needle-trials", type=int, default=10)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    from pagekv import patch_model

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    model_vanilla = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    model_paged   = copy.deepcopy(model_vanilla)
    patch_model(model_paged, page_size=args.page_size, top_k_pages=args.top_k_pages)

    rows: list[BenchRow] = []
    for ctx_len in args.ctx_lens:
        print(f"\n-- ctx_len={ctx_len} --")

        row_v = run_experiment(model_vanilla, ctx_len, "vanilla", args.device, args.needle_trials)
        print(f"  vanilla : mem={row_v.peak_memory_gb}GB  lat={row_v.latency_ms_per_tok}ms/tok  needle={row_v.needle_accuracy:.0%}")
        rows.append(row_v)

        config_tag = f"pagekv_p{args.page_size}_k{args.top_k_pages}"
        row_p = run_experiment(model_paged, ctx_len, config_tag, args.device, args.needle_trials)
        print(f"  pagekv  : mem={row_p.peak_memory_gb}GB  lat={row_p.latency_ms_per_tok}ms/tok  needle={row_p.needle_accuracy:.0%}")
        rows.append(row_p)

    # write CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    print(f"\nCSV saved -> {csv_path}")

    plot_results(rows, out_dir / "memory_vs_context.png")


if __name__ == "__main__":
    main()
