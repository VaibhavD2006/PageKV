from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SearchResult:
    text: str
    score: float
    chunk_id: int
    page_id: int


@dataclass
class DiagnosticResult:
    results: list[SearchResult]
    page_scores: list[float]       # one score per page, length = n_pages
    pages_selected: list[int]      # sorted page indices that were searched
    candidates_searched: list[int] # chunk IDs within the selected pages
