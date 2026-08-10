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

### One-Line Page Reduction (`page_reduce`)

The easiest way to use PageKV. Ask questions over 100K+ token contexts in one line:

```python
from pagekv import page_reduce
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
documents = load_your_large_corpus()  # 100K+ tokens

answer = page_reduce("What are the Mars climate findings?", documents, model)
print(answer)  # Model reads only 5% of context per step
```

`page_reduce` automatically chooses the right router (DynamicPageRouter) for your accuracy needs:
- `accuracy="fast"` — fastest, narrow retrieval
- `accuracy="balanced"` — recommended default
- `accuracy="high"` — closest to vanilla recall at 1M+ tokens

### Patch a HuggingFace model (advanced)

For fine-grained control over routing behavior:

```python
from transformers import AutoModelForCausalLM
from pagekv import patch_model, DynamicPageRouter

model = AutoModelForCausalLM.from_pretrained("MODEL_NAME")

# Always read 5% of pages at every decode step
router = DynamicPageRouter(target_pct=0.05, min_top_k=8)
patch_model(model, page_size=128, router=router)

# use model.generate(...) as normal
```

### Routing diagnostics

`search_with_diagnostics()` exposes the full intermediate routing state so you can separate router failures from within-page failures:

```python
diag = index.search_with_diagnostics(query_embedding, top_k=5)

print(diag.pages_selected)       # page indices the router chose
print(diag.page_scores)          # cosine score for every page (length = n_pages)
print(diag.candidates_searched)  # chunk IDs that were fine-scored
print(diag.results)              # same ranked list as .search()
```

Distinguish two failure modes in evaluation:

```python
gold_chunk_id = 7
gold_page = gold_chunk_id // index.page_size

router_miss = gold_page not in diag.pages_selected
chunk_miss  = gold_chunk_id not in [r.chunk_id for r in diag.results]
gold_page_score = diag.page_scores[gold_page]
```

### ConceptMap — vocabulary gap expansion

Bridges the gap between natural-language queries and technical document vocabulary — e.g. "southern summer" vs "solar longitude 180–360°":

```python
from pagekv import ConceptMap

cm = ConceptMap({
    "southern summer": ["solar longitude 180", "Ls 180-360", "aphelion season"],
    "slope streak":    ["RSL", "recurring slope lineae", "dark streak"],
})
cm.save("mars_concepts.json")
cm = ConceptMap.from_json("mars_concepts.json")

# Returns mean L2-normalized embedding across all expansions
expanded_emb = cm.expand_and_embed("southern summer", embed_one)
results = index.search(expanded_emb, top_k=5)
```

Combine with `search_with_diagnostics()` to measure exactly how much vocabulary expansion improves routing scores:

```python
plain_diag    = index.search_with_diagnostics(embed_one("southern summer"), top_k=5)
expanded_diag = index.search_with_diagnostics(
    cm.expand_and_embed("southern summer", embed_one), top_k=5
)
# Compare plain_diag.page_scores[gold_page] vs expanded_diag.page_scores[gold_page]
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

### LlamaIndex integration (optional)

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

# Accuracy-first — recommended for 100K+ context
router = DynamicPageRouter(target_pct=0.05, min_top_k=8)
patch_model(model, page_size=128, router=router)
```

**Real benchmark results** — CPU attention kernel, page_size=128, B=1 H=12 head_dim=64, 5 warmup + 20 timed runs:

| Context | Vanilla | Fixed k=4 | Dynamic 1% | Dynamic 2% | Dynamic 5% ★ |
|---------|---------|-----------|-----------|-----------|--------------|
| 8K   | 2.402ms | 2.169ms (1.11×) | 1.596ms (1.50×) | 2.529ms (0.95×) | 2.145ms (1.12×) |
| 16K  | 6.186ms | 1.442ms (4.29×) | 3.704ms (1.67×) | 1.636ms (3.78×) | 4.435ms (1.39×) |
| 32K  | 11.761ms | 3.165ms (3.72×) | 1.515ms (7.76×) | 1.957ms (6.01×) | 4.399ms (2.67×) |
| 64K  | 27.149ms | 2.556ms (10.62×) | 2.551ms (10.64×) | 4.694ms (5.78×) | 12.582ms (2.16×) |
| 128K | 54.111ms | 7.442ms (7.27×) | 4.403ms (12.29×) | 11.768ms (4.60×) | 20.817ms (2.60×) |
| 256K | 106.695ms | 3.900ms (27.36×) | 11.444ms (9.32×) | 16.621ms (6.42×) | 37.750ms (2.83×) |

**Context coverage per decode step:**

| Context | Fixed k=4 | Dynamic 5% ★ |
|---------|-----------|--------------|
| 8K   | 512 tokens (6.25%)  | 1,024 tokens (12.50%) |
| 64K  | 512 tokens (0.78%)  | 3,328 tokens (5.08%)  |
| 256K | 512 tokens (0.20%)  | 13,184 tokens (5.03%) |

Fixed `top_k=4` reads 512 tokens always — 6.25% at 8K but only 0.20% at 256K, causing recall to degrade as context grows. Dynamic 5% maintains consistent 5% coverage at every length. GPU benchmarks in progress.

**Choosing `target_pct`:**
- `0.01` — speed-first, narrow retrieval where the answer is in one location
- `0.02` — balanced, good for most long-context workloads
- `0.05` — **accuracy-first, recommended for 100K+ context** where answers may span multiple sections
- `0.05–0.10` — multi-hop reasoning where relevant information is spread across the full context

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

