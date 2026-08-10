# PageKV Evaluation Support: Routing Diagnostics & Vocabulary Gap Expansion

Two new features added to PageKV specifically to support the LUMEN evaluation study. Both are available in the current release via `pip install pagekv`.

---

## Background

PageKV's `Index` retrieves documents in two stages:

1. **Page routing** — the query is compared against a summary vector for each *page* (a group of N chunks). Only the top-scoring pages advance.
2. **Fine-grained search** — within the selected pages, every individual chunk is scored and the best are returned.

The evaluation question your boss raised is whether failures happen at stage 1 (router never selects the right page) or stage 2 (right page selected, wrong chunk wins). Until now, `Index.search()` returned only the final results — the intermediate routing state was invisible. The first feature exposes it. The second feature addresses the vocabulary gap that causes stage 1 to fail in the first place.

---

## Feature 1: Routing Diagnostics

### The problem

When a query like "southern summer" returns the wrong documents, you cannot tell from the results alone *where* it went wrong. Did the router score the relevant page poorly (stage 1 failure), or did it select the right page but pick the wrong chunk within it (stage 2 failure)? These two failure modes call for different fixes — so distinguishing them is essential for a controlled evaluation.

### What it does

`Index.search_with_diagnostics()` runs the exact same two-stage search as `Index.search()` but returns all intermediate routing state alongside the final results.

### API

```python
from pagekv import Index

diag = index.search_with_diagnostics(query_embedding, top_k=5)
```

Returns a `DiagnosticResult` with four fields:

| Field | Type | What it contains |
|---|---|---|
| `results` | `list[SearchResult]` | Same ranked results as `.search()` |
| `page_scores` | `list[float]` | One cosine score per page in the index (length = total pages) |
| `pages_selected` | `list[int]` | The page indices the router chose to search (length = `top_k_pages`) |
| `candidates_searched` | `list[int]` | The chunk IDs that were fine-scored within selected pages |

### Full example — LUMEN evaluation setup

```python
import numpy as np
from pagekv import Index

# ── Build index ──────────────────────────────────────────────────────────────
# 32 chunks, 512-dim CLIP embeddings (HiRISE corpus, small split)
embeddings = np.load("hirise_clip.npy")           # shape [32, 512], L2-normalized
texts       = open("hirise_paths.txt").read().splitlines()  # one image path per line

index = Index.from_embeddings(
    embeddings, texts,
    page_size=8,       # 8 chunks per page → 4 pages total for 32 chunks
    top_k_pages=2,     # router selects 2 pages → 16 candidates fine-scored
)

# ── Query ────────────────────────────────────────────────────────────────────
query_emb = embed("southern summer")   # your CLIP text encoder

diag = index.search_with_diagnostics(query_emb, top_k=5)

# ── Interpret routing ────────────────────────────────────────────────────────
print("All page scores:", [f"{s:.3f}" for s in diag.page_scores])
# e.g. [0.412, 0.381, 0.298, 0.177]  — are the relevant pages scoring high?

print("Pages router selected:", diag.pages_selected)
# e.g. [0, 1]  — did the router pick the pages that contain gold chunks?

print("Chunks fine-scored:", diag.candidates_searched)
# e.g. [0,1,2,...,15]  — all chunks in pages 0 and 1

print("Top result:", diag.results[0].text, f"score={diag.results[0].score:.3f}")
```

### How to use it in the evaluation

For each query with a known gold chunk:

```python
gold_chunk_id = 7   # the chunk that should be returned

diag = index.search_with_diagnostics(query_emb, top_k=5)

gold_page = gold_chunk_id // index._page_size   # which page contains the gold chunk?

# Stage 1 failure: router didn't select the gold page
stage1_failed = gold_page not in diag.pages_selected

# Stage 2 failure: router selected the right page but gold chunk not in top-k results
stage2_failed = (not stage1_failed) and gold_chunk_id not in [r.chunk_id for r in diag.results]

# Page score for the gold page specifically
gold_page_score = diag.page_scores[gold_page]
```

This gives you exactly what your boss described: a clean separation between "wrong page" and "right page, wrong chunk" failures, plus the raw score of the gold page to quantify *how badly* the router missed it.

---

## Feature 2: ConceptMap — Vocabulary Gap Expansion

### The problem

"Southern summer" and "solar longitude 180–360°" do not sit close together in CLIP embedding space. When you embed the query "southern summer" and compare it against page summaries built from text containing "Ls 180" or "solar longitude," the cosine scores are low — not because the content is irrelevant, but because the vocabulary is different. The router may never select the right page regardless of how chunks are organized.

A `ConceptMap` lets you register these equivalences once. Before the query hits the index, it is expanded to include the technical synonyms, and all expansions are embedded and averaged together. The resulting query vector sits closer to the relevant pages in embedding space.

### What it does

`ConceptMap` is a JSON-backed dictionary that maps natural-language terms to their domain equivalents. The key operation is `expand_and_embed`: it embeds every expansion of your query and returns their L2-normalized mean — a single vector that "knows" about both the colloquial and technical forms.

### API

```python
from pagekv import ConceptMap

cm = ConceptMap(mapping)
cm.save(path)                              # persist to JSON
cm = ConceptMap.from_json(path)            # reload

terms = cm.expand(query)                   # returns [query] + synonyms
emb   = cm.expand_and_embed(query, embed) # mean embedding, L2-normalized
```

### Building a concept map for LUMEN

```python
from pagekv import ConceptMap

mars_seasons = ConceptMap({
    "southern summer": ["solar longitude 180", "Ls 180-360", "aphelion season", "Ls 180"],
    "northern summer": ["solar longitude 0",   "Ls 0-180",   "perihelion season"],
    "global dust storm": ["tau > 1", "opacity event", "planet-encircling dust"],
    "dust devil":        ["convective vortex", "PBL vortex"],
    "slope streak":      ["RSL", "recurring slope lineae", "dark streak"],
})

mars_seasons.save("lumen_concept_map.json")   # commit this alongside your dataset
```

### Querying with concept expansion

```python
from pagekv import Index, ConceptMap
import numpy as np

index       = Index.from_embeddings(embeddings, texts, page_size=8, top_k_pages=2)
concept_map = ConceptMap.from_json("lumen_concept_map.json")

# embed_one: callable(str) -> np.ndarray[float32, D]
expanded_emb = concept_map.expand_and_embed("southern summer", embed_one)

# Now search with the enriched embedding
diag = index.search_with_diagnostics(expanded_emb, top_k=5)
```

What happens internally:

```
query = "southern summer"
→ expand() → ["southern summer", "solar longitude 180", "Ls 180-360", "aphelion season", "Ls 180"]
→ embed each → 5 vectors of shape [512]
→ mean → [512]
→ L2-normalize → [512]  ← this is what hits the router
```

The resulting vector is pulled toward "solar longitude" in embedding space without losing the "southern summer" signal.

---

## Complete LUMEN Evaluation Workflow

This is the end-to-end setup for the three-corpus-size evaluation your boss described (32 → ~200 → ~1,600 chunks).

```python
import numpy as np
from pagekv import Index, ConceptMap

# ── Configuration ─────────────────────────────────────────────────────────────
PAGE_SIZE    = 8     # small — chunks are short, one-concept-each
TOP_K_PAGES  = 4
TOP_K        = 5
GOLD_LABELS  = {     # query string → gold chunk ID (your annotation)
    "southern summer":  7,
    "dust devil season": 19,
    "RSL activity":     3,
}

# ── Load data ─────────────────────────────────────────────────────────────────
embeddings = np.load("hirise_clip_32.npy")  # [32, 512]
texts      = open("hirise_paths_32.txt").read().splitlines()
index      = Index.from_embeddings(embeddings, texts, page_size=PAGE_SIZE, top_k_pages=TOP_K_PAGES)

concept_map = ConceptMap.from_json("lumen_concept_map.json")

# ── Evaluate each query ───────────────────────────────────────────────────────
results_log = []

for query_text, gold_chunk_id in GOLD_LABELS.items():
    # Condition A: plain embedding (baseline)
    plain_emb = embed_one(query_text)
    plain_diag = index.search_with_diagnostics(plain_emb, top_k=TOP_K)

    # Condition B: concept-expanded embedding
    expanded_emb = concept_map.expand_and_embed(query_text, embed_one)
    expanded_diag = index.search_with_diagnostics(expanded_emb, top_k=TOP_K)

    gold_page = gold_chunk_id // PAGE_SIZE

    for label, diag in [("plain", plain_diag), ("expanded", expanded_diag)]:
        hit_at_k    = gold_chunk_id in [r.chunk_id for r in diag.results]
        router_miss = gold_page not in diag.pages_selected
        gold_page_score = diag.page_scores[gold_page]

        results_log.append({
            "query":           query_text,
            "condition":       label,
            "hit@k":           hit_at_k,
            "router_miss":     router_miss,
            "gold_page_score": gold_page_score,
            "pages_selected":  diag.pages_selected,
        })

# ── Summarize ─────────────────────────────────────────────────────────────────
import json
print(json.dumps(results_log, indent=2))
```

---

## Installation

```bash
pip install pagekv
```

Both features are included in the current release. No additional dependencies required.

```python
from pagekv import Index, DiagnosticResult, ConceptMap
```

---

## Summary

| Feature | Solves | Key method |
|---|---|---|
| `search_with_diagnostics()` | Can't tell if router failed or fine-search failed | Returns page scores, selected pages, candidate chunk IDs alongside results |
| `ConceptMap` | "Southern summer" ≠ "solar longitude" in embedding space | Expands query to domain synonyms before embedding, returns L2-normalized mean |

Together they let you run the evaluation your boss described: measure where retrieval breaks (router vs. within-page), and measure how much vocabulary expansion closes the gap — at 32, ~200, and ~1,600 chunks.
