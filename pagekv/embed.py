"""Embedding utilities for pagekv.

Provides a thin wrapper around sentence-transformers so users can go
from raw text to a searchable Index in one step.
"""
from __future__ import annotations
import numpy as np


class SentenceTransformerEmbedder:
    """Thin wrapper around a sentence-transformers model.

    Satisfies the embedding_function protocol expected by PageKVRetriever:
        embedding_function(texts: list[str]) -> list[list[float]]

    Usage::

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        index = pagekv.Index.from_embeddings(embedder.embed(texts), texts)

        # Or with LangChain adapter:
        retriever = PageKVRetriever.from_texts(texts, embedder)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install with: pip install pagekv"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns [N, D] float32, L2-normalised (dot-product == cosine similarity)."""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Returns [D] float32 for a single string."""
        return self.embed([text])[0]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts).tolist()

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder(model='{self._model_name}')"
