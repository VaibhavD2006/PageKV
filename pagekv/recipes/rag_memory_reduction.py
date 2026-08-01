#!/usr/bin/env python3
"""RAG with 20x KV cache memory reduction using PageKV.

Use Case: RAG pipelines suffering FROM the very KV cache they need.

# pip install pagekv transformers torch
# python -m pagekv.recipes.rag_memory_reduction \
#     --model meta-llama/Llama-2-7b-chat-hf \
#     --query "What did the Mars rover find?" \
#     --knowledge-base knowledge_base.txt

# Before PageKV: 100K tokens = OOM at 7B model
# After PageKV: 100K tokens = cheap inference
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def chunk_text(text: str, max_tokens: int = 1024) -> list[str]:
    """Naive chunking - replace with your own logic for production."""
    words = text.split()
    chunks = []
    chunk_words = []
    for word in words:
        chunk_words.append(word)
        if len(" ".join(chunk_words)) > 1500:  # rough estimate
            chunks.append(" ".join(chunk_words))
            chunk_words = []
    if chunk_words:
        chunks.append(" ".join(chunk_words))
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load knowledge base
    print(f"Reading knowledge base...", file=sys.stderr)
    kb_text = Path(args.knowledge_base).read_text(encoding="utf-8")
    chunks = chunk_text(kb_text)
    print(f"Split into {len(chunks)} chunks...", file=sys.stderr)

    # RAG loop: retrieve relevant chunks, then generate
    from pagekv import Index, SentenceTransformerEmbedder

    # Build index (one-time cost)
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    embeddings = embedder.embed(chunks)
    index = Index.from_embeddings(embeddings, chunks, page_size=128, top_k_pages=4)

    # Query the index
    query_emb = embedder.embed_one(args.query)
    results = index.search(query_emb, top_k=3)

    # Build prompt with retrieved passages
    context = "\n\n---\n\n".join(r.text for r in results)
    prompt = f"""Answer this question using only the context provided.

Context:
{context}

Question: {args.query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate with PageKV patch
    from pagekv import patch_model, DynamicPageRouter

    router = DynamicPageRouter(target_pct=0.05, min_top_k=8)
    patch_model(model, page_size=128, router=router)

    output = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        do_sample=False,
    )

    print("\n=== ANSWER ===\n")
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove prompt from output
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    print(answer)


if __name__ == "__main__":
    main()