"""
needle_in_haystack.py — measures whether the model attends to a needle token.

Synthetic task: inject a unique token ID at a random position in a long
token sequence, then ask the model to predict the next token after a
"query" marker. Accuracy = fraction of runs where top-1 prediction == needle.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NeedleConfig:
    ctx_len: int
    page_size: int
    top_k_pages: int
    config_name: str  # e.g. "vanilla" or "pagekv_p128_k4"
    n_trials: int = 20
    needle_token_id: int = 999  # arbitrary rare token
    query_token_id: int = 1000


def generate_needle_sequence(
    ctx_len: int,
    needle_token_id: int,
    query_token_id: int,
    vocab_size: int = 50257,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, int]:
    """Return (input_ids [1, ctx_len], needle_position).

    Layout: [random tokens ... needle ... random tokens ... query_marker]
    The model should predict needle_token_id after the query marker.
    """
    rng = random.Random(seed)
    needle_pos = rng.randint(0, ctx_len - 3)

    tokens = [rng.randint(0, vocab_size - 1) for _ in range(ctx_len - 1)]
    tokens[needle_pos] = needle_token_id
    tokens.append(query_token_id)  # query marker at end

    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0), needle_pos


def run_needle_benchmark(
    model: object,
    config: NeedleConfig,
    device: str = "cpu",
) -> float:
    """Return fraction of trials where top-1 prediction matches the needle token."""
    model.eval()
    correct = 0

    for trial in range(config.n_trials):
        input_ids, _ = generate_needle_sequence(
            ctx_len=config.ctx_len,
            needle_token_id=config.needle_token_id,
            query_token_id=config.query_token_id,
            seed=trial,
        )
        input_ids = input_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids).logits  # [1, ctx_len, vocab]
        pred = logits[0, -1, :].argmax().item()
        if pred == config.needle_token_id:
            correct += 1

    return correct / config.n_trials
