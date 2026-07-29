"""Tests for LlamaIndex retriever adapter — llama-index-core mocked."""
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

DIM = 16


class FakeEmbedModel:
    def get_text_embedding_batch(self, texts):
        return [np.random.default_rng(7).standard_normal(DIM).tolist() for _ in texts]

    def get_query_embedding(self, text):
        return np.random.default_rng(7).standard_normal(DIM).tolist()


@pytest.fixture(autouse=True)
def mock_llama(monkeypatch):
    class FakeQueryBundle:
        def __init__(self, qs): self.query_str = qs

    class FakeTextNode:
        def __init__(self, text, id_, metadata=None):
            self.text = text; self.id_ = id_; self.metadata = metadata or {}
        def get_content(self): return self.text

    class FakeNodeWithScore:
        def __init__(self, node, score): self.node = node; self.score = score

    class FakeBaseRetriever:
        def retrieve(self, qs): return self._retrieve(FakeQueryBundle(qs))

    fake = MagicMock()
    fake.BaseRetriever = FakeBaseRetriever
    fake.NodeWithScore = FakeNodeWithScore
    fake.TextNode = FakeTextNode
    fake.QueryBundle = FakeQueryBundle
    for mod in ["llama_index", "llama_index.core", "llama_index.core.retrievers", "llama_index.core.schema"]:
        monkeypatch.setitem(sys.modules, mod, fake)
    with patch(
        "pagekv.integrations.llamaindex_retriever._lazy_llama",
        return_value=(FakeBaseRetriever, FakeNodeWithScore, FakeTextNode, FakeQueryBundle),
    ):
        yield


def test_from_nodes_builds_retriever():
    from pagekv.integrations.llamaindex_retriever import PageKVNodeRetriever
    class FakeNode:
        def get_content(self): return "hello"
    assert PageKVNodeRetriever.from_nodes([FakeNode() for _ in range(20)], FakeEmbedModel(), page_size=5, top_k_pages=2) is not None


def test_retrieve_returns_nodes_with_score():
    from pagekv.integrations.llamaindex_retriever import PageKVNodeRetriever
    class FakeNode:
        def __init__(self, i): self._t = f"node {i}"
        def get_content(self): return self._t
    r = PageKVNodeRetriever.from_nodes([FakeNode(i) for i in range(25)], FakeEmbedModel(), page_size=5, top_k_pages=3, top_k=5).retrieve("q")
    assert len(r) <= 5 and all(hasattr(x, "node") and isinstance(x.score, float) for x in r)


def test_from_index_wraps_existing():
    from pagekv.index import Index
    from pagekv.integrations.llamaindex_retriever import PageKVNodeRetriever
    idx = Index.from_embeddings(
        np.random.randn(20, DIM).astype(np.float32),
        [f"chunk {i}" for i in range(20)],
        page_size=5, top_k_pages=2,
    )
    r = PageKVNodeRetriever.from_index(idx, FakeEmbedModel(), top_k=3).retrieve("q")
    assert isinstance(r, list) and len(r) <= 3


def test_missing_llamaindex_raises():
    with patch("pagekv.integrations.llamaindex_retriever._lazy_llama", side_effect=ImportError("pip install pagekv[llamaindex]")):
        from pagekv.integrations.llamaindex_retriever import PageKVNodeRetriever
        with pytest.raises(ImportError, match="pagekv"):
            PageKVNodeRetriever.from_nodes([], FakeEmbedModel())
