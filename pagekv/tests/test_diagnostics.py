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


# ── ConceptMap tests ──────────────────────────────────────────────────────────

def test_concept_map_importable():
    from pagekv import ConceptMap
    assert ConceptMap is not None


def test_concept_map_expand_exact_match():
    from pagekv import ConceptMap
    cm = ConceptMap({"southern summer": ["solar longitude 180", "Ls 180"]})
    expanded = cm.expand("southern summer")
    assert "southern summer" in expanded
    assert "solar longitude 180" in expanded
    assert "Ls 180" in expanded


def test_concept_map_expand_case_insensitive():
    from pagekv import ConceptMap
    cm = ConceptMap({"Southern Summer": ["solar longitude 180"]})
    expanded = cm.expand("southern summer")
    assert "solar longitude 180" in expanded


def test_concept_map_expand_no_match_returns_original():
    from pagekv import ConceptMap
    cm = ConceptMap({"other term": ["synonym"]})
    expanded = cm.expand("southern summer")
    assert expanded == ["southern summer"]


def test_concept_map_save_and_load(tmp_path):
    from pagekv import ConceptMap
    cm = ConceptMap({"southern summer": ["solar longitude 180", "Ls 180"]})
    path = str(tmp_path / "seasons.json")
    cm.save(path)
    loaded = ConceptMap.from_json(path)
    assert loaded.expand("southern summer") == cm.expand("southern summer")


def test_concept_map_add():
    from pagekv import ConceptMap
    cm = ConceptMap()
    cm.add("northern winter", ["Ls 0", "perihelion"])
    assert "Ls 0" in cm.expand("northern winter")


def test_concept_map_expand_and_embed_returns_vector():
    from pagekv import ConceptMap
    import numpy as np
    cm = ConceptMap({"dust storm": ["tau > 1", "opacity event"]})

    call_log = []
    def fake_embedder(text: str) -> np.ndarray:
        call_log.append(text)
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(8).astype(np.float32)
        return v / np.linalg.norm(v)

    emb = cm.expand_and_embed("dust storm", fake_embedder)
    assert emb.shape == (8,)
    assert len(call_log) == 3  # original + 2 expansions
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5  # L2-normalized
