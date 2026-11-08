#!/usr/bin/env python3
"""
Calibration of the coverage threshold tau: distribution of nearest-neighbour
cosine similarity within the annotated corpus, with covered_thr (p25) and
novel_thr (p10) marked.

Computed directly from the embedding cache with plain numpy (no GPU, no
sentence-transformers), so it runs anywhere within seconds.

Usage
-----
  python3 tau_calibration_plot.py --vecs-cache embeddings_a03ce33c.npy --out-dir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_embeddings(vecs_cache: Path) -> np.ndarray:
    vecs = np.load(str(vecs_cache)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)       # unit length, so dot product = cosine


def top1_without_self(vecs: np.ndarray, batch: int = 2048) -> np.ndarray:
    """For every vector, cosine similarity to its nearest neighbour within the
    same set, excluding the self-match (same logic as in the coverage analysis)."""
    n = len(vecs)
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = vecs[start:end]
        sims = chunk @ vecs.T
        for local_i, global_i in enumerate(range(start, end)):
            sims[local_i, global_i] = -1.0
        out[start:end] = sims.max(axis=1)
    return out


def plot_calibration(
    calib_nn: np.ndarray,
    covered_thr: float,
    novel_thr: float,
    out_path: Path,
    n_bins: int = 60,
) -> dict:
    p10 = float(np.quantile(calib_nn, 0.10))
    p25 = float(np.quantile(calib_nn, 0.25))
    p50 = float(np.quantile(calib_nn, 0.50))
    mean = float(np.mean(calib_nn))
    share_dup = float(np.mean(calib_nn >= 0.99))

    fig, ax = plt.subplots(figsize=(3.6, 2.8), facecolor="white")
    ax.hist(calib_nn, bins=n_bins, range=(0.4, 1.0), color="#9FB3C8", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvline(covered_thr, color="#B5475B", lw=1.6, label=f"\u03c4 = p25 = {covered_thr:.3f}")
    ax.axvline(novel_thr, color="#D9A441", lw=1.4, ls="--", label=f"p10 = {novel_thr:.3f}")
    ax.set_xlabel("Cosine similarity to the nearest neighbour")
    ax.set_ylabel("Examples (log scale)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.text(0.04, 0.50, f"{share_dup:.0%} of examples\nhave a neighbour\nwith similarity \u2265 0.99",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")
    print(f"[stat] share of examples with nearest-neighbour similarity >= 0.99: {share_dup:.3f}")
    return {"mean": mean, "median": p50, "p10": p10, "p25": p25, "share_nn_ge_0.99": share_dup}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs-cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--covered-thr", type=float, default=0.8429,
                    help="Working coverage threshold (default 0.8429, as used in the pipeline)")
    ap.add_argument("--novel-thr", type=float, default=None,
                    help="Novelty threshold; if omitted, p10 of the recomputed distribution is used")
    ap.add_argument("--n-bins", type=int, default=60)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    vecs = load_embeddings(args.vecs_cache)
    print(f"[data] {len(vecs):,} vectors, dim={vecs.shape[1]}")

    print("[calc] Computing nearest-neighbour similarity within the corpus…")
    calib_nn = top1_without_self(vecs)

    novel_thr = args.novel_thr if args.novel_thr is not None else float(np.quantile(calib_nn, 0.10))

    stats = plot_calibration(
        calib_nn, args.covered_thr, novel_thr,
        args.out_dir / "tau_calibration.png", n_bins=args.n_bins,
    )

    # Check how close the stated covered_thr is to p25 of the real distribution
    diff_to_p25 = abs(args.covered_thr - stats["p25"])
    diff_to_p10 = abs(args.covered_thr - stats["p10"])
    closest = "p25" if diff_to_p25 < diff_to_p10 else "p10"

    result = {
        "n_samples": len(vecs),
        "used_covered_thr": args.covered_thr,
        "used_novel_thr": novel_thr,
        "recomputed_stats": stats,
        "covered_thr_closest_to": closest,
    }
    with (args.out_dir / "tau_calibration_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
