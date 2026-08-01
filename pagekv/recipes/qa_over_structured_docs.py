#!/usr/bin/env python3
"""Q&A over structured documents with PageKV memory reduction.

Use Case: Answer questions over thousands of documents without OOM errors.

# pip install pagekv transformers
# python -m pagekv.recipes.qa_over_structured_docs

# Usage:
#   python -m pagekv.recipes.qa_over_structured_docs \
#       --model mistralai/Mistral-7B-Instruct-v0.2 \
#       --docs doc1.txt doc2.txt doc3.txt

# Result: 1-line import, models 100K+ tokens without KV OOM
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--docs", nargs="+", required=True, help="Document files to search")
    parser.add_argument("--query", required=True, help="Question to answer")
    args = parser.parse_args()

    # Load model + tokenizer
    print(f"Loading {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load documents
    print(f"Loading {len(args.docs)} documents...", file=sys.stderr)
    documents = []
    for path in args.docs:
        text = Path(path).read_text(encoding="utf-8")
        documents.append(text)

    # Answer using PageKV's smart routing
    # This is the magic line — handles 100K+ token contexts without OOM
    from pagekv import page_reduce

    answer = page_reduce(
        query=args.query,
        documents=documents,
        model=model,
        accuracy="balanced",  # 5% of context, 2-5x faster
    )

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()