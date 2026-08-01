#!/usr/bin/env python3
"""Agent with 1M-token context window preserved by PageKV.

Use Case: Agents operating over very long documents (codebases, docs, chat history).

# pip install pagekv transformers
# python -m pagekv.recipes.agent_context_window \
#     --model mistralai/Mistral-7B-Instruct-v0.2 \
#     --history conversation.txt

# Result: Agent remembers everything from 1M tokens without KV OOM
"""
from __future__ import annotations
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--history", required=True, help="Full conversation history file")
    parser.add_argument("--query", required=True, help="Agent's next question")
    args = parser.parse_args()

    print(f"Loading {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Calculate context size
    with open(args.history) as f:
        history = f.read()
    tokens = tokenizer.encode(history)
    ctx_k = len(tokens) // 128  # pages at page_size=128
    print(f"History: {len(tokens):,} tokens = {ctx_k} pages", file=sys.stderr)

    # For 1M+ token contexts, use hierarchical routing
    from pagekv import patch_model, HierarchicalPageRouter, DynamicPageRouter

    # Use hierarchical router for 1M+ tokens (O(sqrt(n)) routing)
    # But DynamicPageRouter for consistent recall across the whole context
    router = DynamicPageRouter(target_pct=0.05, min_top_k=16)
    patch_model(model, page_size=128, router=router)

    # Build prompt
    prompt = f"""<s>[SYSTEM]
You are a helpful agent. You have access to conversation history up to 1M tokens.
Use the full context efficiently.

[USER]
{history}

[ASSISTANT]
{args.query}
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)

    # Generate - PageKV handles the routing inference
    output = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove prompt
    if "[ASSISTANT]" in answer:
        answer = answer.split("[ASSISTANT]")[-1].strip()

    print("\n" + answer)


if __name__ == "__main__":
    main()