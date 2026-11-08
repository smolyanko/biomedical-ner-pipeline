#!/usr/bin/env python3
"""
Export the HDBSCAN cluster labels exactly as the training pipeline computes them
(train_ner_cluster_aware.cluster_from_embeddings), one label per example in the
order of the embedding cache (-1 = noise).

The visualisation scripts accept the result via --labels-cache, so the cluster
figures show the very clustering that defines the train/validation/test split.

Usage
-----
  python3 export_cluster_labels.py --data_path train.json \
      --vecs_cache embeddings.npy --umap_cache umap50_cache.npy \
      --out cluster_labels.npy
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import train_ner_cluster_aware as tnc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--vecs_cache", type=Path, required=True)
    ap.add_argument("--umap_cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("cluster_labels.npy"))
    ap.add_argument("--min_cluster_size", type=int, default=30)
    ap.add_argument("--min_samples", type=int, default=5)
    args = ap.parse_args()

    log = logging.getLogger("labels")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    clean = tnc.clean_examples(tnc.load_json(Path(args.data_path).expanduser()))
    clustered = tnc.cluster_from_embeddings(
        clean, embed_model="kamalkraj/BioSimCSE-BioLinkBERT-BASE", embed_device=None, embed_batch=64,
        vecs_cache=args.vecs_cache.expanduser(), logger=log, umap_components=50, umap_neighbors=15,
        umap_min_dist=0.0, umap_cache=args.umap_cache.expanduser() if args.umap_cache else None,
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
        cluster_selection_method="eom")
    labels = np.array([-1 if e["meta"]["cluster"] == tnc.NOISE_CLUSTER_ID else e["meta"]["cluster"]
                       for e in clustered], dtype=int)
    np.save(args.out, labels)
    n_cl = int(labels.max()) + 1
    print(f"saved {args.out}: {len(labels)} examples, {n_cl} clusters, "
          f"{int((labels == -1).sum())} noise ({np.mean(labels == -1):.1%})")


if __name__ == "__main__":
    main()
