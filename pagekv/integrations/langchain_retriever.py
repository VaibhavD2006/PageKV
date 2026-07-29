"""LangChain BaseRetriever adapter for pagekv.Index.

Install with: pip install pagekv[langchain]
"""
from __future__ import annotations
from typing import Any, Callable, List
import numpy as np
from pagekv.index import Index, SearchResult


def _lazy_langchain():
    try:
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        return BaseRetriever, Document, CallbackManagerForRetrieverRun
    except ImportError as exc:
        raise ImportError(
            "LangChain integration requires langchain-core. "
            "Install with: pip install pagekv[langchain]"
        ) from exc


class PageKVRetriever:
    """LangChain-compatible retriever backed by pagekv.Index.

    Usage::

        from pagekv.integrations.langchain_retriever import PageKVRetriever

        retriever = PageKVRetriever.from_texts(
            texts=my_texts,
            embedding_function=openai_embed_fn,   # list[str] -> list[list[float]]
            page_size=128,
            top_k_pages=4,
            top_k=10,
        )
        docs = retriever.get_relevant_documents("my query")
    """

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding_function: Callable[[list[str]], list[list[float]]],
        page_size: int = 128,
        top_k_pages: int = 4,
        top_k: int = 10,
    ) -> Any:
        embeddings = np.array(embedding_function(texts), dtype=np.float32)
        index = Index.from_embeddings(embeddings, texts, page_size=page_size, top_k_pages=top_k_pages)
        return cls._build_retriever(index, embedding_function, top_k)

    @classmethod
    def from_documents(
        cls,
        documents: list[Any],
        embedding_function: Callable[[list[str]], list[list[float]]],
        page_size: int = 128,
        top_k_pages: int = 4,
        top_k: int = 10,
    ) -> Any:
        texts = [doc.page_content for doc in documents]
        return cls.from_texts(texts, embedding_function, page_size, top_k_pages, top_k)

    @classmethod
    def from_index(
        cls,
        index: Index,
        embedding_function: Callable[[list[str]], list[list[float]]],
        top_k: int = 10,
    ) -> Any:
        return cls._build_retriever(index, embedding_function, top_k)

    @staticmethod
    def _build_retriever(index: Index, embedding_function: Callable, top_k: int) -> Any:
        BaseRetriever, Document, CallbackManagerForRetrieverRun = _lazy_langchain()

        # ponytail: closures avoid descriptor protocol eating the callables as unbound methods
        _idx = index
        _embed = embedding_function
        _k = top_k

        class _Retriever(BaseRetriever):
            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> List[Document]:
                query_emb = np.array(_embed([query])[0], dtype=np.float32)
                return [
                    Document(
                        page_content=r.text,
                        metadata={"score": r.score, "chunk_id": r.chunk_id, "page_id": r.page_id},
                    )
                    for r in _idx.search(query_emb, top_k=_k)
                ]

        return _Retriever()
