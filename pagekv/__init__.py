"""
pagekv — page-local KV cache compression for long-context LLM decoding.
"""
from pagekv.integrations.hf_patch import patch_model
from pagekv.core.router import PageRouter, HierarchicalPageRouter, DynamicPageRouter
from pagekv.core.summarizer import (
    BaseSummarizer,
    MeanPoolSummarizer,
    MaxPoolSummarizer,
    LearnedSummarizer,
    train_summarizer,
)
from pagekv.memory.tiering import PageCache
from pagekv.index import Index, SearchResult, DiagnosticResult
from pagekv.embed import SentenceTransformerEmbedder
from pagekv.api import page_reduce
from pagekv.concept_map import ConceptMap

__all__ = [
    "patch_model",
    "PageRouter",
    "HierarchicalPageRouter",
    "DynamicPageRouter",
    "BaseSummarizer",
    "MeanPoolSummarizer",
    "MaxPoolSummarizer",
    "LearnedSummarizer",
    "train_summarizer",
    "PageCache",
    "Index",
    "SearchResult",
    "DiagnosticResult",
    "SentenceTransformerEmbedder",
    "page_reduce",
    "ConceptMap",
]
