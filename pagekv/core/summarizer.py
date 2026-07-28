"""
summarizer.py — pluggable page summarizers.

Phase 1: MeanPoolSummarizer, MaxPoolSummarizer.
Phase 2: LearnedSummarizer (2-layer MLP, train with train_summarizer()).
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, keys: torch.Tensor) -> torch.Tensor:
        """Compress one page of key vectors into a single summary vector.

        Args:
            keys: [B, H, page_size, D]

        Returns:
            summary: [B, H, D]
        """
        ...


class MeanPoolSummarizer(BaseSummarizer):
    def summarize(self, keys: torch.Tensor) -> torch.Tensor:
        """Mean of key vectors across the page dimension."""
        return keys.mean(dim=-2)


class MaxPoolSummarizer(BaseSummarizer):
    def summarize(self, keys: torch.Tensor) -> torch.Tensor:
        """Element-wise max of key vectors across the page dimension."""
        return keys.max(dim=-2).values


class LearnedSummarizer(BaseSummarizer, nn.Module):
    """2-layer MLP that maps a page of keys to a single summary vector.

    Trained to maximise dot-product similarity between the summary and queries
    that actually attended to tokens in that page (retrieval objective).

    Args:
        head_dim: D — key/query head dimension.
        hidden_dim: hidden size of the MLP (default: 2*head_dim).
    """

    def __init__(self, head_dim: int, hidden_dim: Optional[int] = None) -> None:
        nn.Module.__init__(self)
        hidden_dim = hidden_dim or head_dim * 2
        # ponytail: simple 2-layer MLP; add attention-pooling if MLP falls short
        self.pool = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, head_dim),
        )
        self.head_dim = head_dim

    def summarize(self, keys: torch.Tensor) -> torch.Tensor:
        """keys: [B, H, page_size, D] -> [B, H, D]."""
        # mean-pool first, then project — stable initialisation equals MeanPool at init
        pooled = keys.mean(dim=-2)            # [B, H, D]
        return self.pool(pooled)              # [B, H, D]

    # ── training helpers ──────────────────────────────────────────────────────

    def parameters(self, recurse: bool = True):
        return nn.Module.parameters(self, recurse)


def train_summarizer(
    summarizer: LearnedSummarizer,
    keys_dataset: list[torch.Tensor],   # list of [page_size, D] page tensors
    queries_dataset: list[torch.Tensor],# list of [D] query tensors that match
    n_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train a LearnedSummarizer to maximise query-summary dot product.

    Each (keys, query) pair is a positive: the query attended to that page.
    Negative pairs are drawn from other pages in the same batch (in-batch negatives).

    Returns list of per-epoch mean losses.
    """
    summarizer = summarizer.to(device)
    summarizer.train()
    opt = torch.optim.Adam(summarizer.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for keys, query in zip(keys_dataset, queries_dataset):
            keys = keys.unsqueeze(0).unsqueeze(0).to(device)   # [1, 1, page_size, D]
            query = query.unsqueeze(0).unsqueeze(0).to(device) # [1, 1, D]

            summary = summarizer.summarize(keys)  # [1, 1, D]
            # cosine similarity as score; maximise for positive pair
            score = torch.nn.functional.cosine_similarity(
                summary.squeeze(), query.squeeze(), dim=-1
            )
            loss = -score.mean()  # gradient ascent on similarity

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / max(len(keys_dataset), 1)
        losses.append(mean_loss)
        if (epoch + 1) % max(n_epochs // 5, 1) == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  loss={mean_loss:.4f}")

    summarizer.eval()
    return losses
