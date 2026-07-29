"""Tests for SentenceTransformerEmbedder — skipped if sentence-transformers not installed."""
import pytest

st = pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")


def test_embed_shape():
    from pagekv.embed import SentenceTransformerEmbedder
    r = SentenceTransformerEmbedder().embed(["hello world", "foo bar"])
    assert r.shape == (2, 384) and r.dtype.kind == "f"


def test_embed_one_1d():
    from pagekv.embed import SentenceTransformerEmbedder
    r = SentenceTransformerEmbedder().embed_one("hello")
    assert r.ndim == 1 and r.shape[0] == 384


def test_callable_protocol():
    from pagekv.embed import SentenceTransformerEmbedder
    r = SentenceTransformerEmbedder()(["a", "b"])
    assert isinstance(r, list) and isinstance(r[0], list) and isinstance(r[0][0], float)


def test_unit_vectors():
    from pagekv.embed import SentenceTransformerEmbedder
    import numpy as np
    np.testing.assert_allclose(
        np.linalg.norm(SentenceTransformerEmbedder().embed(["test"]), axis=1), 1.0, atol=1e-5
    )


def test_missing_dep_raises():
    import sys
    from unittest.mock import patch
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(ImportError, match=r"pip install pagekv\[embed\]"):
            from pagekv.embed import SentenceTransformerEmbedder
            SentenceTransformerEmbedder()


def test_e2e_with_index():
    from pagekv.embed import SentenceTransformerEmbedder
    from pagekv import Index
    texts = [
        "The capital of France is Paris.",
        "PyTorch is a deep learning framework.",
        "Dogs are mammals.",
    ]
    e = SentenceTransformerEmbedder()
    idx = Index.from_embeddings(e.embed(texts), texts, page_size=2, top_k_pages=2)
    assert "Paris" in idx.search(e.embed_one("What is the capital of France?"), top_k=1)[0].text
