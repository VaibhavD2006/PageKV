"""LlamaIndex BaseRetriever adapter for pagekv.Index."""
from __future__ import annotations
from typing import Any, List
import numpy as np
from pagekv.index import Index


def _lazy_llama():
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
        return BaseRetriever, NodeWithScore, TextNode, QueryBundle
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex integration requires llama-index-core. "
            "Install with: pip install pagekv[llamaindex]"
        ) from exc


class PageKVNodeRetriever:
    """LlamaIndex-compatible retriever backed by pagekv.Index.

    Usage::

        from pagekv.integrations.llamaindex_retriever import PageKVNodeRetriever

        retriever = PageKVNodeRetriever.from_nodes(
            nodes=my_nodes,
            embed_model=embed_model,
            page_size=128,
            top_k_pages=4,
            top_k=10,
        )
        nodes = retriever.retrieve("my query")
    """

    @classmethod
    def from_nodes(
        cls,
        nodes: list[Any],
        embed_model: Any,
        page_size: int = 128,
        top_k_pages: int = 4,
        top_k: int = 10,
    ) -> Any:
        _lazy_llama()  # fail fast if llama-index-core not installed
        texts = [n.get_content() for n in nodes]
        embeddings = np.array(embed_model.get_text_embedding_batch(texts), dtype=np.float32)
        return cls._build_retriever(
            Index.from_embeddings(embeddings, texts, page_size=page_size, top_k_pages=top_k_pages),
            embed_model,
            top_k,
        )

    @classmethod
    def from_index(cls, index: Index, embed_model: Any, top_k: int = 10) -> Any:
        return cls._build_retriever(index, embed_model, top_k)

    @staticmethod
    def _build_retriever(index: Index, embed_model: Any, top_k: int) -> Any:
        BaseRetriever, NodeWithScore, TextNode, QueryBundle = _lazy_llama()

        class _Retriever(BaseRetriever):
            _index = index
            _embed_model = embed_model
            _top_k = top_k

            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                q = np.array(
                    self._embed_model.get_query_embedding(query_bundle.query_str),
                    dtype=np.float32,
                )
                return [
                    NodeWithScore(
                        node=TextNode(
                            text=r.text,
                            id_=str(r.chunk_id),
                            metadata={"page_id": r.page_id},
                        ),
                        score=r.score,
                    )
                    for r in self._index.search(q, top_k=self._top_k)
                ]

        return _Retriever()
