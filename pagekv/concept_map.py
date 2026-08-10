from __future__ import annotations
import json
from pathlib import Path

import numpy as np


class ConceptMap:
    """Synonym/alias expansion map for bridging vocabulary gaps before embedding search.

    Usage::

        cm = ConceptMap({
            "southern summer": ["solar longitude 180", "Ls 180-360", "aphelion season"],
        })
        cm.save("mars_seasons.json")

        terms = cm.expand("southern summer")
        # ["southern summer", "solar longitude 180", "Ls 180-360", "aphelion season"]

        emb = cm.expand_and_embed("southern summer", embedder.embed_one)
    """

    def __init__(self, mapping: dict[str, list[str]] | None = None) -> None:
        self._map: dict[str, list[str]] = {}
        if mapping:
            for k, v in mapping.items():
                self._map[k.lower()] = list(v)

    @classmethod
    def from_json(cls, path: str) -> ConceptMap:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(self._map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, query_term: str, expansions: list[str]) -> None:
        self._map[query_term.lower()] = list(expansions)

    def expand(self, query: str) -> list[str]:
        """Return [query] + all registered expansions. Case-insensitive exact match."""
        synonyms = self._map.get(query.lower(), [])
        return [query] + synonyms

    def expand_and_embed(self, query: str, embedder) -> np.ndarray:
        """Embed all expansions of query and return their L2-normalized mean.

        Args:
            query: natural-language query string
            embedder: callable(str) -> np.ndarray[float32, shape=(D,)]

        Returns:
            L2-normalized mean embedding, shape (D,)
        """
        terms = self.expand(query)
        vectors = np.stack([np.asarray(embedder(t), dtype=np.float32) for t in terms])
        mean_vec = vectors.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec
