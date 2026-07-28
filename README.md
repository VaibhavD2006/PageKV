# PageKV

Page-local compact key summaries for efficient long-context LLM decoding.

> KV cache memory grows linearly with context length. PageKV groups tokens into pages, summarizes each page into a single compact vector, and routes attention to only the most relevant pages — cutting memory and latency at long context with a small, measured accuracy tradeoff.

---

## The Problem

Every time a transformer generates a token, it runs **attention**: the new token's query vector is compared against every key vector ever seen in the context. Both the memory cost (storing all those K/V pairs) and the compute cost (all those dot products) scale **linearly with context length**.

At 100K tokens you're doing 100K dot products and holding 100K key+value vectors in GPU VRAM on every single decode step, for every layer, for every head. That's why long-context inference is memory-bound, not compute-bound — the GPU is mostly waiting for data.

---

## How PageKV Solves It

Most of those 100K tokens are irrelevant to the current query. PageKV exploits this in four steps:

### Step 1 — Paging
Group the KV cache into fixed-size pages (e.g. 128 tokens/page). A 100K-token context becomes ~781 pages.

```
[tok_0 ... tok_127]   → page_0
[tok_128 ... tok_255] → page_1
...
[tok_99968 ... tok_99999] → page_780
```

### Step 2 — Summarise
Compress each page to a single vector by averaging the key vectors (mean pool). The optional learned summarizer (Phase 2) trains a small MLP to do this better.

```
page_0 keys [128, head_dim] → summary_0 [head_dim]
page_1 keys [128, head_dim] → summary_1 [head_dim]
...
781 pages → 781 summary vectors  (always resident in GPU VRAM — tiny)
```

### Step 3 — Route
Dot-product the current query against all 781 summaries. Pick the top-K highest-scoring pages.

```
query · summary_0   = 0.12
query · summary_1   = 0.87  ← selected
query · summary_780 = 0.91  ← selected
...
top-4 pages: [1, 203, 417, 780]
```

### Step 4 — Attend
Load only those 4 pages' raw K/V tensors (4 × 128 = 512 tokens instead of 100K). Run full attention over just 512 tokens.

```
Full attention over 512 tokens  instead of  100,000
```

### The Tradeoff
You're making an approximation — if the router picks the wrong pages you miss relevant tokens. The correctness tests verify that at `top_k_pages = total_pages` the output is **numerically identical** to vanilla attention. Only the top-K truncation introduces approximation, and that is the intended, measured tradeoff.

---

## How the HuggingFace Patch Works

```python
# PageKV registers its attention function:
ALL_ATTENTION_FUNCTIONS["pagekv"] = pagekv_attn_fn

# Then redirects every attention layer:
model.config._attn_implementation = "pagekv"
```

Inside every attention layer's `forward()` HuggingFace dispatches:

```python
attn_output = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation](
    self, query, key, value, ...
)
```

By inserting "pagekv" into that registry, every layer calls `paged_attention_forward` instead of vanilla SDPA. No weights change. No retraining. The model doesn't know anything happened.

---

## Results

*(Run `pagekv-bench --device cuda --ctx-lens 4096 16384 32768 --model <your-model>` to populate — embed chart and table here once real numbers exist.)*

| Context Length | Config  | Peak Memory (GB) | Latency (ms/tok) | Needle Accuracy |
|----------------|---------|-----------------|------------------|----------------|
| 32K            | Vanilla | —               | —                | —              |
| 32K            | PageKV  | —               | —                | —              |

![benchmark chart](pagekv/benchmarks/output/memory_vs_context.png)

---

## Install

```bash
# Core library
pip install pagekv

# With Gradio demo
pip install "pagekv[demo]"

# With vLLM backend
pip install "pagekv[vllm]"

# Development
pip install "pagekv[dev]"
```

---

## Usage

### HuggingFace (drop-in)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pagekv import patch_model

model = AutoModelForCausalLM.from_pretrained("MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")

patch_model(model, page_size=128, top_k_pages=4)

# use model.generate(...) as normal — PageKV runs transparently
```

### Swap summarizer

```python
from pagekv import patch_model, LearnedSummarizer

patch_model(model, page_size=128, top_k_pages=4, summarizer_cls=LearnedSummarizer)
```

### Two-level (hierarchical) routing

```python
from pagekv import patch_model
from pagekv.core.router import HierarchicalPageRouter
from pagekv.integrations.hf_patch import patch_model

# patch_model uses PageRouter by default; for hierarchical routing
# instantiate directly in attention.py or subclass hf_patch
```

### vLLM

```bash
VLLM_ATTENTION_BACKEND=pagekv python your_vllm_server.py
```

---

## Run benchmarks

```bash
# CPU smoke-test (short context, no GPU needed)
pagekv-bench --device cpu --ctx-lens 256 512 1024 --model gpt2

# GPU full benchmark (requires model with long context support)
pagekv-bench --device cuda --ctx-lens 4096 16384 32768 --model mistralai/Mistral-7B-v0.1
```

Outputs `pagekv/benchmarks/output/results.csv` and `memory_vs_context.png`.

---

## Gradio demo

```bash
python demo/app.py           # local at localhost:7860
python demo/app.py --share   # public Gradio link
```

Paste a long document, adjust page size and top-K, compare vanilla vs PageKV output and timing side by side.

---

## Testing

```bash
pytest pagekv/tests/ -v
```

20 tests across Phase 1 and Phase 2. The critical invariant: `test_full_pages_match_vanilla` — at `top_k = total_pages` the paged output must match vanilla attention within `1e-4`. If this fails, do not trust any benchmark numbers.

---

## Architecture

```
pagekv/
├── core/
│   ├── paging.py        paginate_tensor — splits [B,H,S,D] into pages
│   ├── summarizer.py    BaseSummarizer, MeanPool, MaxPool, LearnedSummarizer
│   ├── router.py        PageRouter (flat), HierarchicalPageRouter (two-level)
│   └── attention.py     paged_attention_forward — orchestrates steps 1-4
├── integrations/
│   ├── hf_patch.py      patch_model — injects pagekv into HF attention dispatch
│   └── vllm_backend.py  PageKVAttentionBackend — vLLM plugin
├── memory/
│   └── tiering.py       PageCache — LRU GPU-hot / CPU-cold KV page cache
├── benchmarks/
│   ├── needle_in_haystack.py   retrieval accuracy measurement
│   ├── memory_profile.py       peak GPU memory + decode latency
│   └── run_all.py              matrix orchestrator → CSV + chart
├── demo/
│   └── app.py           Gradio UI — paste text, compare vanilla vs PageKV
└── tests/
    ├── test_correctness.py    Phase 1 — paging, routing, HF patch
    └── test_phase2.py         Phase 2 — learned summarizer, hierarchical router, tiering
```

### Phase summary

| Phase | Deliverables |
|---|---|
| 1 ✅ | HF patch, mean/max pool, flat routing, benchmarks, correctness tests |
| 2 ✅ | Trainable MLP summarizer, LRU CPU/GPU tiering, two-level hierarchical routing |
| 3 ✅ | vLLM backend plugin, Gradio demo, pip packaging (classifiers, entry points, extras) |

---

## Status

All three phases implemented. GPU benchmark numbers pending — see `docs/prd.md` for success metrics (≥30% memory reduction, ≤5% accuracy drop at 32K+ tokens).