# PageKV

Page-local compact key summaries for efficient long-context LLM decoding — plus a standalone semantic search index.

> KV cache memory grows linearly with context length. PageKV groups tokens into pages, summarizes each page into a single compact vector, and routes attention to only the most relevant pages — cutting memory and latency at long context with a small, measured accuracy tradeoff.

## Install

```bash
pip install pagekv
```

LlamaIndex integration (optional — large dependency):

```bash
pip install pagekv[llamaindex]
```

## Usage

### Patch a HuggingFace model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pagekv import patch_model

model = AutoModelForCausalLM.from_pretrained("MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")

patch_model(model, page_size=128, top_k_pages=4)

# use model.generate(...) as normal — PageKV runs transparently
```

### pagekv.Index — standalone semantic search

`pagekv.Index` is a two-stage retrieval index that runs in-process with no external dependencies. Bring your own embeddings — any model works.

```python
import numpy as np
from pagekv import Index, SearchResult

embeddings = np.load("embeddings.npy")   # [N, D] float32
texts = open("chunks.txt").read().splitlines()

index = Index.from_embeddings(embeddings, texts, page_size=128, top_k_pages=4)

query_emb = embed_model("What is attention?")   # [D] float32
results: list[SearchResult] = index.search(query_emb, top_k=5)

for r in results:
    print(f"[{r.score:.3f}] {r.text[:80]}")

# Save / load (no pickle)
index.save("my_index.npz")
loaded = Index.load("my_index.npz")
```

`SearchResult` fields: `text: str`, `score: float`, `chunk_id: int`, `page_id: int`

### With sentence-transformers

```python
from pagekv import Index
from pagekv.embed import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
texts = ["Paris is the capital of France.", "PyTorch is a deep learning library."]

index = Index.from_embeddings(embedder.embed(texts), texts, page_size=2)
results = index.search(embedder.embed_one("capital of France"), top_k=1)
print(results[0].text)
```

### LangChain integration

```python
from langchain_openai import OpenAIEmbeddings
from pagekv.integrations.langchain_retriever import PageKVRetriever

retriever = PageKVRetriever.from_texts(
    texts=my_chunks,
    embedding_function=OpenAIEmbeddings().embed_documents,
    page_size=128,
    top_k_pages=4,
    top_k=10,
)

docs = retriever.get_relevant_documents("explain attention mechanisms")
```

### LlamaIndex integration

```bash
pip install pagekv[llamaindex]
```

```python
from llama_index.core.node_parser import SentenceSplitter
from pagekv.integrations.llamaindex_retriever import PageKVNodeRetriever

nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(docs)

retriever = PageKVNodeRetriever.from_nodes(
    nodes=nodes,
    embed_model=embed_model,
    page_size=128,
    top_k_pages=4,
    top_k=10,
)

results = retriever.retrieve("explain self-attention")
```

### DynamicPageRouter — constant recall quality at any context length

Fixed `top_k` reads the same number of tokens regardless of context size. At 100K tokens `top_k=4` covers 0.5% of the cache — acceptable. At 1M tokens it covers 0.05% — the model may miss relevant pages entirely.

`DynamicPageRouter` fixes this by reading a fixed **percentage** of pages per decode step, so recall quality stays roughly constant as context grows:

```python
from pagekv import patch_model, DynamicPageRouter

router = DynamicPageRouter(
    target_pct=0.01,   # read 1% of pages every decode step
    min_top_k=4,       # always attend at least 4 pages at short context
)
patch_model(model, page_size=128, router=router)
```

What 1% costs at different context lengths (page_size=128):

| Context | Pages | top_k | Tokens read | vs vanilla |
|---------|-------|-------|------------|------------|
| 100K | 781 | 8 | 1,024 | ~8x faster |
| 500K | 3,906 | 40 | 5,120 | ~40x faster |
| 1M | 7,812 | 79 | 10,112 | ~130x faster |

Compare to fixed `top_k=4` at 1M tokens: 512 tokens = 0.05% of context. `DynamicPageRouter(0.01)` reads 20× more context at the same scale while still being ~130× faster than vanilla.

**Choosing `target_pct`:**
- `0.005` — fastest, use when speed matters more than recall
- `0.01` — recommended for most long-context workloads
- `0.02` — closer to vanilla recall quality, still very fast at 1M+ tokens

### HierarchicalPageRouter — scaling to very long context

At 1M+ tokens with `page_size=128` you have ~7,800 pages. `PageRouter` scans all of them every decode step — O(n_pages). `HierarchicalPageRouter` adds a second routing level that groups pages into super-pages, cutting the scan to O(√n_pages) while returning the same top-k result.

```python
from pagekv import patch_model
from pagekv.core.router import HierarchicalPageRouter

router = HierarchicalPageRouter(top_k=4, super_page_size=32)
patch_model(model, page_size=128, top_k_pages=4, router=router)
```

| | `PageRouter` | `HierarchicalPageRouter` |
|---|---|---|
| Routing cost | O(n_pages) | O(√n_pages) |
| Result quality | baseline | identical at `top_k=total_pages` |
| Best for | up to ~100K tokens | 100K+ tokens |

`super_page_size=32` at 1M tokens: 7,812 pages → 245 super-pages scanned at level 1, then 32 candidates at level 2.

## How it works

Attention over page summaries picks the top-K relevant pages, then full attention runs only within those pages instead of over the entire cache. The same two-stage routing powers both `patch_model` and `pagekv.Index`.

See [`docs/arch.md`](docs/arch.md) for the full design.

## Status

Early-stage MVP. Contributions and issues welcome.

## License

MIT
