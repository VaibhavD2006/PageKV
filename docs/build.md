# ARCHITECTURE: PageKV

## 1. High-Level Concept

Standard attention compares a query against every stored key in the KV
cache (O(n) in sequence length, both in compute and memory). PageKV inserts
an intermediate routing step:

```
Query
  │
  ▼
[1] Compare query against PAGE SUMMARIES (cheap, small set)
  │
  ▼
[2] Select top-K most relevant pages
  │
  ▼
[3] Run full attention only over keys/values in those top-K pages
  │
  ▼
Output
```

This trades a small amount of accuracy (approximate routing) for large
memory and latency savings at long context lengths.

## 2. Module Breakdown

```
pagekv/
├── core/
│   ├── paging.py          # splits token stream into fixed-size pages
│   ├── summarizer.py       # pluggable: MeanPoolSummarizer, MaxPoolSummarizer,
│   │                       #   LearnedSummarizer (Phase 2)
│   ├── router.py           # compares query vs page summaries, returns top-K page ids
│   └── attention.py        # drop-in replacement attention forward function
├── integrations/
│   ├── hf_patch.py         # monkey-patches HuggingFace transformers attention
│   └── vllm_backend.py     # (Phase 3) custom vLLM attention backend
├── memory/
│   └── tiering.py          # (Phase 2) GPU-hot / CPU-cold page eviction logic
├── benchmarks/
│   ├── needle_in_haystack.py
│   ├── memory_profile.py
│   └── run_all.py          # produces the README benchmark table/chart
├── demo/
│   └── app.py               # Gradio demo: paste long doc, compare vanilla vs PageKV
└── tests/
    └── test_correctness.py  # PageKV output ≈ vanilla output on short sequences
```

## 3. Data Flow / Memory Tiering

- **Hot tier (GPU VRAM):** raw K/V tensors for the current page(s) selected
  by the router, plus all page summary vectors (small, always resident).
- **Cold tier (CPU RAM, Phase 2):** raw K/V tensors for pages not recently
  selected, evicted from GPU and reloaded on demand if the router selects
  them again.
- No external database is used. This is a memory-tiering problem, not a
  data-storage problem — page summaries and raw K/V are just tensors held
  in process memory (GPU or CPU), not persisted to disk in the MVP.

## 4. Key Interfaces

```python
# core/summarizer.py
class BaseSummarizer:
    def summarize(self, keys: torch.Tensor) -> torch.Tensor:
        """keys: [page_size, head_dim] -> returns [head_dim] summary vector"""
        raise NotImplementedError

class MeanPoolSummarizer(BaseSummarizer):
    def summarize(self, keys):
        return keys.mean(dim=0)

# core/router.py
class PageRouter:
    def __init__(self, top_k: int):
        self.top_k = top_k

    def select_pages(self, query: torch.Tensor,
                      page_summaries: torch.Tensor) -> torch.Tensor:
        """Returns indices of top_k most relevant pages for this query."""
        scores = query @ page_summaries.T
        return scores.topk(self.top_k).indices

# integrations/hf_patch.py
def patch_model(model, page_size=128, top_k_pages=4,
                 summarizer_cls=MeanPoolSummarizer):
    """Replaces model's attention forward with PageKV attention."""
    ...
```

## 5. Benchmark Harness Design

`benchmarks/run_all.py` must produce, for a matrix of context lengths
(e.g. 4K / 16K / 32K / 64K / 128K tokens):

| Context Length | Config    | Peak Memory (GB) | Decode Latency (ms/tok) | Needle Accuracy (%) |
|-----------------|-----------|-------------------|---------------------------|------------------------|
| 32K             | Vanilla   | ...               | ...                       | ...                    |
| 32K             | PageKV    | ...               | ...                       | ...                    |

This table is the core artifact for the README and for any resume/demo use.
Output should also be rendered as a matplotlib chart (memory vs. context
length, two lines: vanilla vs PageKV).

## 6. Correctness Guarantee (Important)

Before any benchmarking work, `tests/test_correctness.py` must verify: at
`top_k_pages = total_pages` (i.e., no pages excluded), PageKV attention
output must numerically match vanilla attention output within floating
point tolerance. This proves the routing mechanism itself introduces no
bugs — only the top-K truncation introduces approximation, and that's the
intended, measured tradeoff.

## 7. Phase 2/3 Notes (do not build yet, but design interfaces to allow)

- **Learned summarizer:** small linear or 2-layer MLP head trained to
  predict a summary vector that maximizes downstream retrieval accuracy —
  train this after the mean-pool baseline is benchmarked, not before.
- **Hierarchical/recursive routing:** stack multiple levels of page
  summaries (summaries-of-summaries) — keep `PageRouter` interface
  page-list-in, page-list-out so a hierarchical router can be swapped in
  without touching `attention.py`.
- **vLLM backend:** implement as a custom attention backend class per
  vLLM's plugin API — do this only after Phase 1 numbers are solid, since
  vLLM internals are more complex to debug than a HuggingFace patch.