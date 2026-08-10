import numpy as np
import pytest
from pagekv import Index, SearchResult
from pagekv.index.results import DiagnosticResult


def _make_index(n=16, dim=8, page_size=4, top_k_pages=2):
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    texts = [f"chunk_{i}" for i in range(n)]
    return Index.from_embeddings(embs, texts, page_size=page_size, top_k_pages=top_k_pages), embs


def test_diagnostic_result_importable():
    from pagekv.index.results import DiagnosticResult
    assert DiagnosticResult is not None


def test_search_with_diagnostics_returns_diagnostic_result():
    index, embs = _make_index()
    q = embs[0]
    result = index.search_with_diagnostics(q, top_k=3)
    assert isinstance(result, DiagnosticResult)


def test_diagnostic_results_match_plain_search():
    index, embs = _make_index()
    q = embs[0]
    plain = index.search(q, top_k=3)
    diag = index.search_with_diagnostics(q, top_k=3)
    assert len(diag.results) == len(plain)
    for a, b in zip(diag.results, plain):
        assert a.chunk_id == b.chunk_id
        assert abs(a.score - b.score) < 1e-5


def test_page_scores_length_equals_n_pages():
    index, embs = _make_index(n=16, page_size=4)  # 4 pages
    q = embs[0]
    diag = index.search_with_diagnostics(q, top_k=3)
    assert len(diag.page_scores) == 4


def test_pages_selected_count_equals_top_k_pages():
    index, embs = _make_index(n=16, page_size=4, top_k_pages=2)
    q = embs[0]
    diag = index.search_with_diagnostics(q, top_k=3)
    assert len(diag.pages_selected) == 2


def test_candidates_searched_within_selected_pages():
    index, embs = _make_index(n=16, page_size=4, top_k_pages=2)
    q = embs[0]
    diag = index.search_with_diagnostics(q, top_k=3)
    for cid in diag.candidates_searched:
        assert cid // 4 in diag.pages_selected


def test_pages_selected_are_sorted():
    index, embs = _make_index(n=16, page_size=4, top_k_pages=2)
    q = embs[0]
    diag = index.search_with_diagnostics(q, top_k=3)
    assert diag.pages_selected == sorted(diag.pages_selected)
