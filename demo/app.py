"""
demo/app.py — Gradio demo: paste a long document, compare vanilla vs PageKV.

Timing is split into prefill (processing the input context) and decode
(generating each new token). PageKV only compresses the decode phase —
prefill uses standard causal attention. The stats table shows decode
ms/token so the compression benefit is visible even at short context lengths.

Launch: python demo/app.py
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

_loaded: dict = {}


def _get_models(model_name: str):
    if model_name not in _loaded:
        tok = AutoTokenizer.from_pretrained(model_name)
        vanilla = AutoModelForCausalLM.from_pretrained(model_name).eval()
        _loaded[model_name] = {"tokenizer": tok, "vanilla": vanilla}
    return _loaded[model_name]


SUMMARIZER_MAP = {
    "mean": MeanPoolSummarizer,
    "max": MaxPoolSummarizer,
    "learned": LearnedSummarizer,
}


def _run_model(model, ids: torch.Tensor, max_new_tokens: int):
    """Run prefill then greedy decode one token at a time.

    Returns (token_id_list, prefill_ms, decode_ms_per_token).
    Splitting prefill from decode lets us isolate where PageKV actually
    differs: prefill uses vanilla causal SDPA; decode routes each step
    to only the top-K pages of the growing KV cache.
    """
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(ids)
        prefill_ms = (time.perf_counter() - t0) * 1000

        current = ids
        step_times = []
        token_ids = []
        for _ in range(max_new_tokens):
            t0 = time.perf_counter()
            out = model(current)
            step_times.append((time.perf_counter() - t0) * 1000)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_ids.append(next_id.item())
            current = torch.cat([current, next_id], dim=-1)

    decode_ms = sum(step_times) / len(step_times) if step_times else 0.0
    return token_ids, prefill_ms, decode_ms


def run_comparison(
    input_text: str,
    model_name: str,
    page_size: int,
    top_k_pages: int,
    max_new_tokens: int,
    summarizer_name: str,
) -> tuple[str, str, str]:
    if not input_text.strip():
        return "", "", "Paste some text first."

    cache = _get_models(model_name)
    tok = cache["tokenizer"]
    vanilla_model = cache["vanilla"]

    paged_model = copy.deepcopy(vanilla_model)
    summarizer_cls = SUMMARIZER_MAP.get(summarizer_name, MeanPoolSummarizer)
    patch_model(paged_model, page_size=page_size, top_k_pages=top_k_pages,
                summarizer_cls=summarizer_cls)
    paged_model.eval()

    tok.pad_token = tok.eos_token
    ids = tok(input_text, return_tensors="pt").input_ids
    ctx_len = ids.shape[-1]

    max_ctx = getattr(vanilla_model.config, "max_position_embeddings", 1024)
    if ctx_len > max_ctx - max_new_tokens:
        ids = ids[:, -(max_ctx - max_new_tokens):]
        ctx_len = ids.shape[-1]

    v_ids, v_pre, v_dec = _run_model(vanilla_model, ids, max_new_tokens)
    p_ids, p_pre, p_dec = _run_model(paged_model,   ids, max_new_tokens)

    vanilla_text = tok.decode(v_ids, skip_special_tokens=True)
    paged_text   = tok.decode(p_ids, skip_special_tokens=True)

    n_pages     = -(-ctx_len // page_size)
    k_used      = min(top_k_pages, n_pages)
    dec_speedup = v_dec / max(p_dec, 0.001)
    pct_ctx     = round(100 * k_used * page_size / ctx_len)

    stats = (
        f"| | Vanilla | PageKV |\n"
        f"|---|---|---|\n"
        f"| Context tokens | {ctx_len} | {ctx_len} |\n"
        f"| Pages attended (decode) | {n_pages} / {n_pages} (100%) | {k_used} / {n_pages} ({pct_ctx}%) |\n"
        f"| Prefill time | {v_pre:.0f} ms | {p_pre:.0f} ms |\n"
        f"| Decode ms / token | {v_dec:.1f} ms | {p_dec:.1f} ms |\n"
        f"| Decode speedup | 1.00x | {dec_speedup:.2f}x |\n"
        f"\n*Prefill is identical — paging only activates during decode (Sq=1 per step). "
        f"Speedup grows with context length and is most visible on GPU.*"
    )

    return vanilla_text, paged_text, stats


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PageKV Demo") as demo:
        gr.Markdown("# PageKV — KV Cache Compression Demo")
        gr.Markdown(
            "Paste a long document. **Prefill** (processing the full context) is the same "
            "for both models. PageKV compresses the **decode** phase: each new token attends "
            "only to the top-K most relevant pages of the KV cache instead of all of it. "
            "The stats table shows decode ms/token to isolate that benefit."
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
                page_size    = gr.Slider(8, 128, value=64, step=8,  label="Page size (tokens)")
                top_k_pages  = gr.Slider(1, 32,  value=8,  step=1,  label="Top-K pages (decode)")
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
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
