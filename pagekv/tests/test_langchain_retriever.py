"""Tests for LangChain retriever adapter — langchain-core mocked."""
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

DIM = 16


def _fake_embed_fn(texts):
    rng = np.random.default_rng(42)
    return [rng.standard_normal(DIM).tolist() for _ in texts]


@pytest.fixture(autouse=True)
def mock_langchain(monkeypatch):
    class FakeDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class FakeBaseRetriever:
        def get_relevant_documents(self, query):
            return self._get_relevant_documents(query, run_manager=None)

    class FakeCallbackManager:
        pass

    fake = MagicMock()
    fake.BaseRetriever = FakeBaseRetriever
    fake.Document = FakeDocument
    fake.CallbackManagerForRetrieverRun = FakeCallbackManager
    for mod in ["langchain_core", "langchain_core.retrievers", "langchain_core.documents", "langchain_core.callbacks"]:
        monkeypatch.setitem(sys.modules, mod, fake)
    with patch(
        "pagekv.integrations.langchain_retriever._lazy_langchain",
        return_value=(FakeBaseRetriever, FakeDocument, FakeCallbackManager),
    ):
        yield


def test_from_texts_builds_retriever():
    from pagekv.integrations.langchain_retriever import PageKVRetriever
    assert PageKVRetriever.from_texts([f"doc {i}" for i in range(30)], _fake_embed_fn, page_size=8, top_k_pages=3, top_k=5) is not None


def test_from_texts_returns_documents():
    from pagekv.integrations.langchain_retriever import PageKVRetriever
    docs = PageKVRetriever.from_texts([f"doc {i}" for i in range(30)], _fake_embed_fn, page_size=8, top_k_pages=3, top_k=5).get_relevant_documents("q")
    assert isinstance(docs, list) and len(docs) <= 5 and all(hasattr(d, "page_content") for d in docs)


def test_from_texts_document_metadata():
    from pagekv.integrations.langchain_retriever import PageKVRetriever
    docs = PageKVRetriever.from_texts([f"doc {i}" for i in range(20)], _fake_embed_fn, page_size=5, top_k_pages=2, top_k=3).get_relevant_documents("q")
    for doc in docs:
        assert "score" in doc.metadata and "chunk_id" in doc.metadata and "page_id" in doc.metadata


def test_from_documents():
    from pagekv.integrations.langchain_retriever import PageKVRetriever
    class FakeDoc:
        def __init__(self, t): self.page_content = t
    assert len(PageKVRetriever.from_documents([FakeDoc(f"c{i}") for i in range(20)], _fake_embed_fn, page_size=5, top_k_pages=2).get_relevant_documents("q")) > 0


def test_from_index_wraps_existing():
    from pagekv.index import Index
    from pagekv.integrations.langchain_retriever import PageKVRetriever
    embs = np.random.randn(20, DIM).astype(np.float32)
    idx = Index.from_embeddings(embs, [f"chunk {i}" for i in range(20)], page_size=5, top_k_pages=2)
    assert isinstance(PageKVRetriever.from_index(idx, _fake_embed_fn, top_k=3).get_relevant_documents("q"), list)


def test_missing_langchain_raises():
    import importlib, sys
    # Remove cached module to force re-import
    mods = [k for k in sys.modules if "pagekv.integrations.langchain_retriever" in k]
    for m in mods: del sys.modules[m]
    with patch("pagekv.integrations.langchain_retriever._lazy_langchain", side_effect=ImportError("pip install pagekv[langchain]")):
        from pagekv.integrations.langchain_retriever import PageKVRetriever
        with pytest.raises(ImportError, match="pagekv"):
            PageKVRetriever.from_texts(["a"], _fake_embed_fn)
