# PRD: PageKV — Page-Local Compact Key Summaries for Efficient Long-Context Decoding

## 1. Problem Statement

Transformer LLMs store a Key-Value (KV) cache entry for every token they've ever
processed in a conversation or document. Memory usage grows **linearly** with
context length, and attention cost (time spent comparing a new query against
every stored key) grows the same way. This means:

- Long-context inference (100K+ tokens) is memory-bound, not compute-bound.
- Serving providers pay for GPU memory that mostly holds redundant/rarely-used
  key vectors.
- Users hit context-length walls or pay a steep latency/cost tax for long
  documents and long conversations.

There is no widely-available, benchmarked, open-source library that
compresses the KV cache using **page-local summarization** — grouping tokens
into fixed-size pages, computing a compact summary per page, and using those
summaries to route attention to only the relevant pages.

## 2. Goal

Build and open-source **PageKV**: a Python library that plugs into an
existing LLM inference stack (HuggingFace `transformers` for MVP, vLLM as a
stretch goal) and reduces KV cache memory usage on long-context inference
with a measurable, published accuracy/latency/memory tradeoff curve.

## 3. Non-Goals (MVP)

- Not a hosted SaaS product in v1.
- Not a training/fine-tuning project — no new model weights, no learned
  compressor initially (start with a non-learned summary, e.g. mean/max
  pooling).
- Not a RAG or external-document retrieval system — this operates entirely
  inside the model's own attention mechanism.
- Not a general-purpose vector database — GPU/CPU memory tiering only.
- Not initially targeting multi-GPU/distributed serving — single GPU first.

## 4. Target Users

- **ML/infra engineers** at companies serving long-context LLMs who want to
  cut GPU memory cost.
- **Open-source LLM serving projects** (vLLM, TGI) as a potential future
  integration/plugin.
- **The builder's own portfolio**: this is also a resume/demo project, so a
  secondary "user" is a technical hiring manager evaluating the repo.

## 5. Core User Stories

1. As an ML engineer, I can `pip install pagekv` and wrap my existing
   HuggingFace model's attention with a single call, without retraining.
2. As an ML engineer, I can configure page size and summary strategy and see
   memory savings and latency change accordingly.
3. As a builder, I can run a benchmark script that outputs a table/chart of
   memory used, latency, and accuracy (needle-in-haystack recall) for vanilla
   attention vs. PageKV at multiple context lengths.
4. As a public visitor, I can try PageKV in a hosted demo (Colab or HF Space)
   without installing anything.

## 6. Success Metrics (MVP)

- **Memory:** ≥30% reduction in peak KV cache memory at 32K+ token context,
  vs. vanilla full KV cache, at a fixed page size.
- **Accuracy:** ≤5% drop in needle-in-haystack retrieval accuracy vs. vanilla
  baseline, at the memory-saving configuration used above.
- **Latency:** Neutral-to-positive effect on decode latency at long context
  (attention over summaries + top pages should be no slower than full
  attention beyond ~16K tokens).
- **Usability:** A new user can go from `pip install` to a working benchmark
  chart in under 15 minutes, following the README.

## 7. Scope Phasing

- **Phase 1 (MVP):** HuggingFace `transformers` integration, mean-pooled page
  summaries, top-k page routing, benchmark suite, README with charts.
- **Phase 2:** Learned page summarizer (small projection head), hierarchical
  (multi-level / recursive) summary tree, CPU-offload for cold pages.
- **Phase 3:** vLLM custom attention backend / plugin, HF Space demo,
  packaging polish for public pip release.

## 8. Risks / Open Questions

- Mean/max-pooled summaries may not capture enough signal for good page
  routing — may need a lightweight learned summarizer sooner than planned.
  Mitigation: benchmark both, keep the interface pluggable from day one.
- GPU kernels for page-level top-k routing may not parallelize well without
  custom CUDA/Triton work — mitigate by keeping v1 in plain PyTorch and only
  optimizing kernels after correctness is proven.
- Recursive/hierarchical routing adds engineering complexity — explicitly
  deferred to Phase 2 so Phase 1 stays shippable.