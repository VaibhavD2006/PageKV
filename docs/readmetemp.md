# PageKV

Page-local compact key summaries for efficient long-context LLM decoding — plus a standalone semantic search index.

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

Optional extras:

```bash
pip install pagekv[embed]      # sentence-transformers embedder
pip install pagekv[langchain]  # LangChain BaseRetriever adapter
pip install pagekv[llamaindex] # LlamaIndex BaseRetriever adapter
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

`pagekv.Index` is a two-stage retrieval index that runs in-process with no
external dependencies. It works with any embedding model: a coarse page-summary
dot-product selects the top-k most relevant pages, then a fine dot-product
within those pages ranks chunks.

```python
import numpy as np
from pagekv import Index, SearchResult

# Build from any [N, D] float32 embedding array
embeddings = np.load("embeddings.npy")
texts = open("chunks.txt").read().splitlines()

index = Index.from_embeddings(
    embeddings, texts,
    page_size=128,   # chunks per page
    top_k_pages=4,   # coarse-pass page budget
)

# Search returns SearchResult dataclasses
query_emb = embed_model("What is attention?")   # [D] float32
results: list[SearchResult] = index.search(query_emb, top_k=5)

for r in results:
    print(f"[{r.score:.3f}] page={r.page_id} chunk={r.chunk_id}: {r.text[:80]}")

# Persist without pickle
index.save("my_index.npz")
loaded = Index.load("my_index.npz")
```

**SearchResult fields:** `text: str`, `score: float`, `chunk_id: int`, `page_id: int`

**Incremental updates:**

```python
index.add(embedding, text)                    # one chunk
index.add_batch(embeddings_array, texts_list) # many at once
```

### With sentence-transformers (`pagekv[embed]`)

```python
from pagekv import Index
from pagekv.embed import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
texts = ["Paris is the capital of France.", "PyTorch is a deep learning library."]

index = Index.from_embeddings(embedder.embed(texts), texts, page_size=2)
results = index.search(embedder.embed_one("capital of France"), top_k=1)
print(results[0].text)
```

`SentenceTransformerEmbedder` satisfies the `embedding_function` protocol
(`list[str] → list[list[float]]`) expected by the LangChain adapter.

### LangChain integration (`pagekv[langchain]`)

```python
from langchain_openai import OpenAIEmbeddings
from pagekv.integrations.langchain_retriever import PageKVRetriever

embed_fn = OpenAIEmbeddings().embed_documents

retriever = PageKVRetriever.from_texts(
    texts=my_chunks,
    embedding_function=embed_fn,
    page_size=128,
    top_k_pages=4,
    top_k=10,
)

docs = retriever.get_relevant_documents("explain attention mechanisms")

# Wrap an existing pagekv.Index
from pagekv import Index
index = Index.load("my_index.npz")
retriever = PageKVRetriever.from_index(index, embed_fn, top_k=5)

# From LangChain Document objects
retriever = PageKVRetriever.from_documents(lc_docs, embed_fn)
```

### LlamaIndex integration (`pagekv[llamaindex]`)

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from pagekv.integrations.llamaindex_retriever import PageKVNodeRetriever

nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(
    SimpleDirectoryReader("./data").load_data()
)

retriever = PageKVNodeRetriever.from_nodes(
    nodes=nodes,
    embed_model=embed_model,
    page_size=128,
    top_k_pages=4,
    top_k=10,
)

results = retriever.retrieve("explain self-attention")

# Wrap an existing pagekv.Index
from pagekv import Index
index = Index.load("my_index.npz")
retriever = PageKVNodeRetriever.from_index(index, embed_model, top_k=5)
```

## How it works

See `ARCHITECTURE.md` for the full design. Short version: attention over
page summaries picks the top-K relevant pages, then full attention runs
only within those pages instead of over the entire cache.

The same two-stage routing that makes `patch_model` efficient also powers
`pagekv.Index`: page-level dot-product narrows the search space, then
chunk-level dot-product ranks results within those pages.

## Try it live

*(Link to Gradio demo / HuggingFace Space once Phase 5 is complete.)*

## Status

Early-stage, MVP per `PRD.md`. Contributions and issues welcome.

## License

*(Choose: MIT is the standard default for this kind of infra tooling.)*
