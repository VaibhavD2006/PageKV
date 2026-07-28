# PageKV

Page-local compact key summaries for efficient long-context LLM decoding.

> KV cache memory grows linearly with context length. PageKV groups tokens
> into pages, summarizes each page into a single compact vector, and routes
> attention to only the most relevant pages — cutting memory and latency at
> long context with a small, measured accuracy tradeoff.

## Why

Long-context inference is memory-bound, not compute-bound. Every token you
process gets stored as a Key-Value pair that must stay resident for the
rest of the generation. PageKV compresses that cache by summarizing pages
of tokens instead of keeping every token's raw key/value around.

## Results

*(Fill in after Phase 4 of BUILD_PLAN.md — embed the benchmark chart and
table here. Do not publish this README until real numbers exist.)*

| Context Length | Config  | Peak Memory | Latency (ms/tok) | Needle Accuracy |
|-----------------|---------|--------------|---------------------|---------------------|
| 32K             | Vanilla | —            | —                   | —                   |
| 32K             | PageKV  | —            | —                   | —                   |

![benchmark chart](benchmarks/output/memory_vs_context.png)

## Install

```bash
pip install pagekv
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pagekv import patch_model

model = AutoModelForCausalLM.from_pretrained("MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")

patch_model(model, page_size=128, top_k_pages=4)

# use model.generate(...) as normal — PageKV runs transparently
```

## How it works

See `ARCHITECTURE.md` for the full design. Short version: attention over
page summaries picks the top-K relevant pages, then full attention runs
only within those pages instead of over the entire cache.

## Try it live

*(Link to Gradio demo / HuggingFace Space once Phase 5 is complete.)*

## Status

Early-stage, MVP per `PRD.md`. Contributions and issues welcome.

## License

*(Choose: MIT is the standard default for this kind of infra tooling.)*