from dataclasses import dataclass

@dataclass
class SearchResult:
    text: str
    score: float
    chunk_id: int
    page_id: int
