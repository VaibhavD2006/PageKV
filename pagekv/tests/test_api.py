"""Tests for page_reduce() convenience API."""
import pytest


def test_page_reduce_importable():
    """page_reduce should be importable from top-level pagekv package."""
    from pagekv import page_reduce
    assert callable(page_reduce)


def test_page_reduce_signature():
    """page_reduce should accept expected parameters."""
    from pagekv import page_reduce
    import inspect
    sig = inspect.signature(page_reduce)
    params = list(sig.parameters.keys())
    assert "query" in params
    assert "documents" in params
    assert "model" in params
    assert "accuracy" in params