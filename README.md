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

## How it works

Attention over page summaries picks the top-K relevant pages, then full attention runs only within those pages instead of over the entire cache. The same two-stage routing powers both `patch_model` and `pagekv.Index`.

See `ARCHITECTURE.md` for the full design.

## Status

Early-stage MVP. Contributions and issues welcome.

## License

MIT
