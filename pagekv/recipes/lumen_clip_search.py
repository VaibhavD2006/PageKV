#!/usr/bin/env python3
"""LUMEN planetary image search using CLIP embeddings + PageKV.

Use Case: Search HiRISE and Mars mission images by text or image query.
Used in: LUMEN v1 (ICMLA 2025) with NASA JPL Planetary Data System datasets.

# pip install pagekv torch transformers Pillow

## Build and save an index from a directory of images:
# python -m pagekv.recipes.lumen_clip_search \
#     --images /data/hirise/ --save-index hirise.npz

## Query an existing index:
# python -m pagekv.recipes.lumen_clip_search \
#     --load-index hirise.npz --query "dark sand dunes on Mars"

## Add pre-computed embeddings (skip re-embedding):
# python -m pagekv.recipes.lumen_clip_search \
#     --embeddings hirise_clip.npy --image-list paths.txt \
#     --save-index hirise.npz --query "slope streaks"
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


MODEL_NAME = "openai/clip-vit-base-patch32"  # 512-dim; swap for ViT-L/14 for 768-dim


def _load_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def embed_images(paths: list[str], batch_size: int = 32) -> np.ndarray:
    """Return L2-normalized CLIP image embeddings, shape [N, D]."""
    from PIL import Image
    model, processor = _load_clip()
    all_embs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
        print(f"  embedded {min(i + batch_size, len(paths))}/{len(paths)}", file=sys.stderr)
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def embed_text(query: str) -> np.ndarray:
    """Return L2-normalized CLIP text embedding for cross-modal search, shape [D]."""
    model, processor = _load_clip()
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--images", help="Directory of image files to embed and index")
    src.add_argument("--embeddings", help=".npy file of pre-computed CLIP embeddings [N, D]")
    src.add_argument("--load-index", help="Load a previously saved .npz index (skip building)")
    parser.add_argument("--image-list", help="Text file with one image path per line (used with --embeddings)")
    parser.add_argument("--save-index", help="Save built index to .npz for reuse")
    parser.add_argument("--query", required=True, help="Text query for cross-modal image search")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=64,
                        help="Images per page (default 64 — smaller than text since images are semantically diverse)")
    parser.add_argument("--top-k-pages", type=int, default=4)
    parser.add_argument("--exts", default=".jpg,.jpeg,.png,.tif,.tiff",
                        help="Image extensions to scan (comma-separated)")
    args = parser.parse_args()

    from pagekv import Index

    if args.load_index:
        print(f"Loading index from {args.load_index}...", file=sys.stderr)
        index = Index.load(args.load_index)
        print(f"  {index}", file=sys.stderr)

    elif args.embeddings:
        embeddings = np.load(args.embeddings)
        if args.image_list:
            image_paths = Path(args.image_list).read_text(encoding="utf-8").splitlines()
        else:
            parser.error("--embeddings requires --image-list")
        print(f"Building index: {len(image_paths)} images, dim={embeddings.shape[1]}", file=sys.stderr)
        index = Index.from_embeddings(embeddings, image_paths,
                                      page_size=args.page_size, top_k_pages=args.top_k_pages)

    elif args.images:
        exts = set(args.exts.split(","))
        image_paths = sorted(
            str(p) for p in Path(args.images).rglob("*") if p.suffix.lower() in exts
        )
        if not image_paths:
            sys.exit(f"No images found in {args.images} with extensions {exts}")
        print(f"Embedding {len(image_paths)} images...", file=sys.stderr)
        embeddings = embed_images(image_paths)
        print(f"Building index...", file=sys.stderr)
        index = Index.from_embeddings(embeddings, image_paths,
                                      page_size=args.page_size, top_k_pages=args.top_k_pages)

    else:
        parser.error("Provide one of: --images, --embeddings, or --load-index")

    if args.save_index and not args.load_index:
        index.save(args.save_index)
        print(f"Index saved to {args.save_index}", file=sys.stderr)

    print(f"\nSearching: '{args.query}'", file=sys.stderr)
    query_emb = embed_text(args.query)
    results = index.search(query_emb, top_k=args.top_k)

    print(f"\nTop {args.top_k} results for: \"{args.query}\"")
    for i, r in enumerate(results, 1):
        print(f"  {i}. score={r.score:.4f}  {r.text}")


if __name__ == "__main__":
    main()
