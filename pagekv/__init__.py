"""
pagekv — page-local KV cache compression for long-context LLM decoding.
"""
from pagekv.integrations.hf_patch import patch_model
from pagekv.core.router import PageRouter, HierarchicalPageRouter
from pagekv.core.summarizer import (
    BaseSummarizer,
    MeanPoolSummarizer,
    MaxPoolSummarizer,
    LearnedSummarizer,
    train_summarizer,
)
from pagekv.memory.tiering import PageCache
from pagekv.index import Index, SearchResult

__all__ = [
    "patch_model",
    "PageRouter",
    "HierarchicalPageRouter",
    "BaseSummarizer",
    "MeanPoolSummarizer",
    "MaxPoolSummarizer",
    "LearnedSummarizer",
    "train_summarizer",
    "PageCache",
    "Index",
    "SearchResult",
]
