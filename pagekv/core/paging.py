"""
paging.py — splits KV tensors into fixed-size pages and maps page indices back to tokens.
"""
import math
import torch


def paginate_tensor(x: torch.Tensor, page_size: int) -> tuple[torch.Tensor, int]:
    """Split x along the sequence dimension into fixed-size pages.

    Args:
        x: [B, H, S, D]
        page_size: tokens per page

    Returns:
        pages: [B, H, n_pages, page_size, D]  (last page zero-padded if S % page_size != 0)
        real_seq_len: original S before padding
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    B, H, S, D = x.shape
    real_seq_len = S
    n_pages = math.ceil(S / page_size)
    pad = n_pages * page_size - S

    if pad > 0:
        padding = torch.zeros(B, H, pad, D, dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=2)

    pages = x.reshape(B, H, n_pages, page_size, D)
    return pages, real_seq_len


def get_page_token_indices(
    page_indices: torch.Tensor,
    page_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Map selected page indices to flat token positions.

    Args:
        page_indices: [B, H, top_k]  — sorted page indices
        page_size: tokens per page
        seq_len: original sequence length (used to clip out-of-bounds pad tokens)

    Returns:
        token_indices: [B, H, top_k * page_size]  — flat token indices, clamped to [0, seq_len)
    """
    B, H, top_k = page_indices.shape
    # offsets[p] = p * page_size; arange over page_size gives per-page token offsets
    page_offsets = page_indices * page_size  # [B, H, top_k]
    token_offsets = torch.arange(page_size, device=page_indices.device)  # [page_size]
    # broadcast: [B, H, top_k, page_size]
    indices = page_offsets.unsqueeze(-1) + token_offsets
    # flatten and clamp to valid range
    indices = indices.reshape(B, H, top_k * page_size).clamp(0, seq_len - 1)
    return indices
