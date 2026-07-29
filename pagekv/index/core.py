from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
from pagekv.index.results import SearchResult

class Index:
    def __init__(self, embeddings, texts, page_size=128, top_k_pages=4):
        if len(texts) != embeddings.shape[0]:
            raise ValueError(f"embeddings has {embeddings.shape[0]} rows but texts has {len(texts)} entries")
        if page_size <= 0: raise ValueError(f"page_size must be positive, got {page_size}")
        if top_k_pages <= 0: raise ValueError(f"top_k_pages must be positive, got {top_k_pages}")
        if len(texts) == 0: raise ValueError("Index requires at least one embedding")
        self._embeddings = embeddings.float()
        self._texts = list(texts)
        self._page_size = page_size
        self._top_k_pages = top_k_pages
        self._n = len(texts)
        self._n_pages = math.ceil(self._n / page_size)
        self._page_summaries = self._build_summaries()

    @classmethod
    def from_embeddings(cls, embeddings, texts, page_size=128, top_k_pages=4):
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(np.array(embeddings), dtype=torch.float32)
        return cls(embeddings, texts, page_size, top_k_pages)

    def search(self, query_embedding, top_k=10):
        q = torch.as_tensor(
            np.array(query_embedding) if not isinstance(query_embedding, torch.Tensor) else query_embedding,
            dtype=torch.float32,
        )
        if q.dim() == 1: q = q.unsqueeze(0)
        page_scores = torch.matmul(q, self._page_summaries.T).squeeze(0)
        k_pages = min(self._top_k_pages, self._n_pages)
        top_page_indices = page_scores.topk(k_pages).indices.sort().values.tolist()
        candidate_ids = []
        for page_idx in top_page_indices:
            start = page_idx * self._page_size
            end = min(start + self._page_size, self._n)
            candidate_ids.extend(range(start, end))
        if not candidate_ids: return []
        cand_embs = self._embeddings[candidate_ids]
        fine_scores = torch.matmul(q, cand_embs.T).squeeze(0)
        k_return = min(top_k, len(candidate_ids))
        top_vals, top_local = fine_scores.topk(k_return)
        return [
            SearchResult(
                text=self._texts[candidate_ids[i]],
                score=float(s),
                chunk_id=candidate_ids[i],
                page_id=candidate_ids[i] // self._page_size,
            )
            for s, i in zip(top_vals.tolist(), top_local.tolist())
        ]

    def add(self, embedding, text):
        emb = torch.as_tensor(
            np.array(embedding) if not isinstance(embedding, torch.Tensor) else embedding,
            dtype=torch.float32,
        )
        if emb.dim() == 1: emb = emb.unsqueeze(0)
        if emb.shape[0] != 1:
            raise ValueError(f"add() expects a single embedding (shape [D] or [1,D]), got shape {tuple(emb.shape)}")
        self._embeddings = torch.cat([self._embeddings, emb], dim=0)
        self._texts.append(text)
        self._n = len(self._texts)
        self._n_pages = math.ceil(self._n / self._page_size)
        self._page_summaries = self._build_summaries()

    def add_batch(self, embeddings, texts):
        embs = torch.as_tensor(
            np.array(embeddings) if not isinstance(embeddings, torch.Tensor) else embeddings,
            dtype=torch.float32,
        )
        if len(texts) != embs.shape[0]: raise ValueError("embeddings and texts must have the same length")
        self._embeddings = torch.cat([self._embeddings, embs], dim=0)
        self._texts.extend(texts)
        self._n = len(self._texts)
        self._n_pages = math.ceil(self._n / self._page_size)
        self._page_summaries = self._build_summaries()

    def save(self, path):
        path = Path(path)
        texts_bytes = json.dumps(self._texts).encode("utf-8")
        np.savez(
            path,
            embeddings=self._embeddings.numpy(),
            page_summaries=self._page_summaries.numpy(),
            page_size=np.array([self._page_size]),
            top_k_pages=np.array([self._top_k_pages]),
            texts_json=np.frombuffer(texts_bytes, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        texts = json.loads(data["texts_json"].tobytes().decode("utf-8"))
        idx = cls.__new__(cls)
        idx._embeddings = torch.from_numpy(data["embeddings"].copy()).float()
        idx._texts = texts
        idx._page_size = int(data["page_size"][0])
        idx._top_k_pages = int(data["top_k_pages"][0])
        idx._n = len(texts)
        idx._n_pages = math.ceil(idx._n / idx._page_size)
        idx._page_summaries = torch.from_numpy(data["page_summaries"].copy()).float()
        if idx._embeddings.shape[0] != idx._n:
            raise ValueError(f"Corrupted index: embeddings has {idx._embeddings.shape[0]} rows but texts has {idx._n} entries")
        if idx._page_summaries.shape[0] != idx._n_pages:
            raise ValueError(f"Corrupted index: page_summaries has {idx._page_summaries.shape[0]} pages but expected {idx._n_pages}")
        return idx

    def _build_summaries(self):
        summaries = []
        for i in range(self._n_pages):
            start = i * self._page_size
            end = min(start + self._page_size, self._n)
            summaries.append(self._embeddings[start:end].mean(dim=0))
        return torch.stack(summaries)

    def __len__(self): return self._n
    def __repr__(self):
        return f"Index(n={self._n}, dim={self._embeddings.shape[1]}, n_pages={self._n_pages}, page_size={self._page_size}, top_k_pages={self._top_k_pages})"
