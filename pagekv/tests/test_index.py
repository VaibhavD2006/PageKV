import math
import numpy as np
import pytest
import torch
from pagekv.index import Index, SearchResult

DIM = 32
N = 50

def _e(n=N, d=DIM, seed=0): return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
def _t(n=N): return [f"chunk {i}" for i in range(n)]

def test_from_embeddings_numpy():
    idx = Index.from_embeddings(_e(), _t())
    assert len(idx) == N and idx._page_size == 128 and idx._top_k_pages == 4

def test_from_embeddings_torch():
    assert len(Index.from_embeddings(torch.randn(N, DIM), _t())) == N

def test_from_embeddings_list_of_lists():
    assert len(Index.from_embeddings([[float(i)] * DIM for i in range(N)], _t())) == N

def test_from_embeddings_mismatch_raises():
    with pytest.raises(ValueError, match="texts"):
        Index.from_embeddings(_e(n=10), ["only one text"])

def test_invalid_page_size_raises():
    with pytest.raises(ValueError): Index.from_embeddings(_e(), _t(), page_size=0)

def test_invalid_top_k_raises():
    with pytest.raises(ValueError): Index.from_embeddings(_e(), _t(), top_k_pages=0)

def test_page_summary_shape():
    idx = Index.from_embeddings(_e(n=300), _t(n=300), page_size=128)
    assert idx._page_summaries.shape == (math.ceil(300 / 128), DIM)

def test_search_returns_search_results():
    idx = Index.from_embeddings(_e(), _t())
    results = idx.search(_e(n=1)[0], top_k=5)
    assert all(isinstance(r, SearchResult) for r in results) and len(results) <= 5

def test_search_top_k_capped_by_n():
    assert len(Index.from_embeddings(_e(n=3), _t(n=3), page_size=128, top_k_pages=4).search(_e(n=1)[0], top_k=100)) == 3

def test_search_scores_descending():
    scores = [r.score for r in Index.from_embeddings(_e(), _t()).search(_e(n=1)[0], top_k=10)]
    assert scores == sorted(scores, reverse=True)

def test_search_exact_match_ranks_first():
    embs = _e(n=20)
    idx = Index.from_embeddings(embs, _t(n=20), page_size=5, top_k_pages=4)
    assert idx.search(embs[7].copy(), top_k=1)[0].chunk_id == 7

def test_search_with_torch_query():
    assert len(Index.from_embeddings(_e(), _t()).search(torch.tensor(_e(n=1)[0]), top_k=5)) > 0

def test_search_result_fields():
    embs = _e(n=10)
    idx = Index.from_embeddings(embs, _t(n=10), page_size=3, top_k_pages=2)
    r = idx.search(embs[0], top_k=3)[0]
    assert isinstance(r.text, str) and isinstance(r.score, float) and r.page_id == r.chunk_id // 3

def test_add_single():
    idx = Index.from_embeddings(_e(n=10), _t(n=10))
    idx.add(np.random.randn(DIM).astype(np.float32), "new chunk")
    assert len(idx) == 11 and idx._texts[-1] == "new chunk"

def test_add_batch():
    idx = Index.from_embeddings(_e(n=10), _t(n=10))
    idx.add_batch(_e(n=5, seed=99), [f"new {i}" for i in range(5)])
    assert len(idx) == 15

def test_add_batch_mismatch_raises():
    idx = Index.from_embeddings(_e(n=5), _t(n=5))
    with pytest.raises(ValueError): idx.add_batch(_e(n=3), ["only one text"])

def test_add_updates_page_summaries():
    idx = Index.from_embeddings(_e(n=10), _t(n=10), page_size=5)
    old = idx._n_pages
    for i in range(5): idx.add(np.random.randn(DIM).astype(np.float32), f"extra {i}")
    assert idx._n_pages >= old

def test_added_chunk_is_searchable():
    idx = Index.from_embeddings(_e(n=20), _t(n=20), page_size=5, top_k_pages=4)
    distinctive = np.ones(DIM, dtype=np.float32)
    idx.add(distinctive, "distinctive chunk")
    assert idx.search(distinctive, top_k=1)[0].text == "distinctive chunk"

def test_save_and_load(tmp_path):
    embs, texts = _e(), _t()
    idx = Index.from_embeddings(embs, texts, page_size=16, top_k_pages=2)
    idx.save(tmp_path / "test.npz")
    loaded = Index.load(tmp_path / "test.npz")
    assert loaded._page_size == 16 and len(loaded) == N and loaded._texts == texts
    assert torch.allclose(loaded._embeddings, idx._embeddings)

def test_save_load_search_consistency(tmp_path):
    embs = _e()
    idx = Index.from_embeddings(embs, _t(), page_size=16, top_k_pages=3)
    original = [r.chunk_id for r in idx.search(embs[5], top_k=5)]
    idx.save(tmp_path / "idx.npz")
    loaded = Index.load(tmp_path / "idx.npz")
    assert original == [r.chunk_id for r in loaded.search(embs[5], top_k=5)]

def test_save_no_pickle(tmp_path):
    idx = Index.from_embeddings(_e(n=5), _t(n=5))
    idx.save(tmp_path / "idx.npz")
    data = np.load(tmp_path / "idx.npz")
    assert "embeddings" in data and "texts_json" in data

def test_repr():
    idx = Index.from_embeddings(_e(), _t())
    assert "Index" in repr(idx) and str(N) in repr(idx)

def test_top_level_import():
    import pagekv
    assert hasattr(pagekv, "Index") and hasattr(pagekv, "SearchResult")

def test_empty_index_raises():
    with pytest.raises(ValueError):
        Index.from_embeddings(np.zeros((0, DIM), dtype=np.float32), [])

def test_add_2d_batch_raises():
    idx = Index.from_embeddings(_e(n=5), _t(n=5))
    with pytest.raises(ValueError):
        idx.add(np.random.randn(2, DIM).astype(np.float32), "text")

def test_add_batch_searchable():
    idx = Index.from_embeddings(_e(n=10), _t(n=10), page_size=5, top_k_pages=2)
    target = np.ones(DIM, dtype=np.float32)
    idx.add_batch(
        np.stack([target, np.zeros(DIM, dtype=np.float32)]),
        ["target chunk", "zero chunk"],
    )
    results = idx.search(target, top_k=1)
    assert results[0].text == "target chunk"
