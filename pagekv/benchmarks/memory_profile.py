"""
memory_profile.py — measures peak GPU/CPU memory and decode latency.
"""
from __future__ import annotations
import time
import warnings

import torch


def measure_peak_memory_gb(
    model: object,
    input_ids: torch.Tensor,
    device: str,
) -> float:
    """Peak memory (GB) allocated during a single forward pass.

    Returns 0.0 on CPU with a warning — CPU allocators don't expose peak usage.
    """
    if device == "cpu" or not torch.cuda.is_available():
        warnings.warn("Peak memory measurement requires CUDA; returning 0.0 on CPU.")
        with torch.no_grad():
            model(input_ids)
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(input_ids.to(device))
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 3)


def measure_decode_latency_ms(
    model: object,
    input_ids: torch.Tensor,
    n_new_tokens: int = 10,
    device: str = "cpu",
    n_warmup: int = 1,
) -> float:
    """Mean latency per generated token (ms) over n_new_tokens steps.

    Uses model.generate() with greedy decoding and no KV cache reuse between
    timing runs to isolate per-token latency.
    """
    input_ids = input_ids.to(device)
    model.eval()

    # warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=2, do_sample=False)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=n_new_tokens, do_sample=False)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms / n_new_tokens
