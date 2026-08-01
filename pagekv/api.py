"""High-level convenience API for scientists and agents.

page_reduce provides a one-line interface for RAG-style queries
over long-context documents, using PageKV's dynamic routing under the hood.
"""
from __future__ import annotations
from typing import Literal, Optional
import warnings


def page_reduce(
    query: str,
    documents: list[str],
    model,
    accuracy: Literal["fast", "balanced", "high"] = "balanced",
    page_size: int = 128,
) -> str:
    """Reduce a long context to the most relevant portion for a query.

    This is a convenience wrapper around PageKV's routing capabilities.
    It attends only to a percentage of your context while maintaining
    high recall quality — perfect for RAG, Q&A over long docs, or
    agents operating over 100K+ token contexts.

    Args:
        query: The question or search string
        documents: List of text chunks to search through
        model: HuggingFace PreTrainedModel with a tokenizer
            (or a model with .tokenizer attribute)
        accuracy: Tradeoff between speed and recall quality
            - "fast": target_pct=0.01 (fastest, narrow retrieval)
            - "balanced": target_pct=0.05 (recommended default)
            - "high": target_pct=0.10 (closest to vanilla recall)
        page_size: Tokens per page (default 128, match patch_model defaults)

    Returns:
        Generated answer string from the model

    Example:
        >>> from pagekv import page_reduce
        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        >>> docs = load_large_document_corpus()  # 100K+ tokens
        >>> answer = page_reduce("What are the climate findings?", docs, model)

    Note:
        For maximum speed with acceptable recall, use accuracy="balanced".
        For scientific/analytical tasks requiring broad recall, use accuracy="high".
    """
    from pagekv import patch_model, DynamicPageRouter

    # Map accuracy tiers to target_pct (fraction of pages searched)
    pct_map = {"fast": 0.01, "balanced": 0.05, "high": 0.10}
    target_pct = pct_map[accuracy]

    # Ensure minimum coverage at short contexts
    min_top_k = 4 if accuracy == "fast" else 8

    router = DynamicPageRouter(target_pct=target_pct, min_top_k=min_top_k)

    # Get tokenizer
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        # Try to infer from model config
        model_name = getattr(model.config, "_name_or_path", None)
        if model_name is None:
            raise ValueError(
                "model must have a .tokenizer attribute or a .config._name_or_path "
                "that can be used to load a tokenizer. "
                "Try: tokenizer = AutoTokenizer.from_pretrained('model_name')"
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare inputs
    prompt = f"<s>[INST] {query} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device if hasattr(model, "device") else "cpu")

    # Apply PageKV patch with dynamic router
    patch_model(model, page_size=page_size, router=router)

    # Generate answer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress tokenizer/device warnings
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True if accuracy != "fast" else False,
            temperature=0.7 if accuracy != "fast" else 0.0,
        )

    # Decode and return
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)