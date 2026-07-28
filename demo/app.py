"""
demo/app.py — Gradio demo: paste a long document, compare vanilla vs PageKV.

Launch: python demo/app.py
Or share publicly: python demo/app.py --share
"""
from __future__ import annotations
import copy
import time
import argparse

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

from pagekv import patch_model
from pagekv.core.summarizer import MeanPoolSummarizer, MaxPoolSummarizer, LearnedSummarizer

# ── model cache (load once per session) ───────────────────────────────────────

_loaded: dict = {}


def _get_models(model_name: str):
    if model_name not in _loaded:
        tok = AutoTokenizer.from_pretrained(model_name)
        vanilla = AutoModelForCausalLM.from_pretrained(model_name).eval()
        _loaded[model_name] = {"tokenizer": tok, "vanilla": vanilla}
    return _loaded[model_name]


# ── inference ─────────────────────────────────────────────────────────────────

SUMMARIZER_MAP = {
    "mean": MeanPoolSummarizer,
    "max": MaxPoolSummarizer,
    "learned": LearnedSummarizer,
}


def run_comparison(
    input_text: str,
    model_name: str,
    page_size: int,
    top_k_pages: int,
    max_new_tokens: int,
    summarizer_name: str,
) -> tuple[str, str, str]:
    """Returns (vanilla_output, pagekv_output, stats_markdown)."""
    if not input_text.strip():
        return "", "", "Paste some text first."

    cache = _get_models(model_name)
    tok = cache["tokenizer"]
    vanilla_model = cache["vanilla"]

    # Build patched model fresh each call (config mutation is cheap)
    paged_model = copy.deepcopy(vanilla_model)
    summarizer_cls = SUMMARIZER_MAP.get(summarizer_name, MeanPoolSummarizer)
    patch_model(paged_model, page_size=page_size, top_k_pages=top_k_pages,
                summarizer_cls=summarizer_cls)
    paged_model.eval()

    tok.pad_token = tok.eos_token
    ids = tok(input_text, return_tensors="pt").input_ids
    ctx_len = ids.shape[-1]

    # cap to model max context
    max_ctx = getattr(vanilla_model.config, "max_position_embeddings", 1024)
    if ctx_len > max_ctx - max_new_tokens:
        ids = ids[:, -(max_ctx - max_new_tokens):]
        ctx_len = ids.shape[-1]

    def generate(model, label):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        new_ids = out[0, ctx_len:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        return text, elapsed_ms

    vanilla_text, vanilla_ms = generate(vanilla_model, "vanilla")
    paged_text,   paged_ms   = generate(paged_model,   "pagekv")

    n_pages = -(-ctx_len // page_size)  # ceil div
    stats = (
        f"| | Vanilla | PageKV |\n"
        f"|---|---|---|\n"
        f"| Context tokens | {ctx_len} | {ctx_len} |\n"
        f"| KV pages | {n_pages} | top-{min(top_k_pages, n_pages)} of {n_pages} |\n"
        f"| Generation time | {vanilla_ms:.0f} ms | {paged_ms:.0f} ms |\n"
        f"| Speedup | 1.00x | {vanilla_ms/max(paged_ms,1):.2f}x |\n"
    )

    return vanilla_text, paged_text, stats


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PageKV Demo") as demo:
        gr.Markdown("# PageKV — KV Cache Compression Demo")
        gr.Markdown(
            "Paste a long document below. PageKV routes attention to only the "
            "most relevant pages instead of the full KV cache."
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Input text (paste anything long here)",
                    lines=10,
                    placeholder="Paste a long document, article, or conversation...",
                )
            with gr.Column(scale=1):
                model_name   = gr.Dropdown(["gpt2", "gpt2-medium"], value="gpt2", label="Model")
                page_size    = gr.Slider(8, 128, value=32, step=8, label="Page size (tokens)")
                top_k_pages  = gr.Slider(1, 16, value=4, step=1, label="Top-K pages")
                max_new_toks = gr.Slider(10, 100, value=30, step=10, label="Max new tokens")
                summarizer   = gr.Radio(["mean", "max", "learned"], value="mean", label="Summarizer")
                run_btn      = gr.Button("Run comparison", variant="primary")

        stats_md = gr.Markdown(label="Stats")

        with gr.Row():
            vanilla_out = gr.Textbox(label="Vanilla output", lines=5)
            paged_out   = gr.Textbox(label="PageKV output", lines=5)

        run_btn.click(
            fn=run_comparison,
            inputs=[input_text, model_name, page_size, top_k_pages, max_new_toks, summarizer],
            outputs=[vanilla_out, paged_out, stats_md],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
