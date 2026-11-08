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
    return vecs / np.maximum(norms, 1e-9)


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

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")

    ax.hist(calib_nn, bins=n_bins, color="#A8D8EA", edgecolor="white",
             linewidth=0.4, zorder=2)

    ymax = ax.get_ylim()[1]

    # novel_thr (p10)
    ax.axvline(novel_thr, color="#F7A9A8", linestyle="--", linewidth=2, zorder=3)
    ax.text(novel_thr, ymax * 0.97, f" novel_thr = p10 = {novel_thr:.4f}",
            rotation=90, va="top", ha="right", fontsize=9, color="#C0605F")

    # covered_thr (p25) — the working threshold
    ax.axvline(covered_thr, color="#5B9279", linestyle="-", linewidth=2.2, zorder=4)
    ax.text(covered_thr, ymax * 0.97, f" covered_thr = p25 = {covered_thr:.4f}",
            rotation=90, va="top", ha="right", fontsize=9, color="#3C6B54",
            fontweight="bold")

    # median — for context
    ax.axvline(p50, color="#999999", linestyle=":", linewidth=1.3, zorder=3)
    ax.text(p50, ymax * 0.97, f" median = {p50:.4f}",
            rotation=90, va="top", ha="right", fontsize=8.5, color="#777777")

    # percentile labels below the x axis
    for q_val, q_label, color in [(p10, "10%", "#C0605F"),
                                    (p25, "25%", "#3C6B54"),
                                    (p50, "50%", "#777777")]:
        ax.annotate(
            q_label, xy=(q_val, 0), xytext=(q_val, -ymax * 0.06),
            ha="center", va="top", fontsize=8, color=color,
            annotation_clip=False,
        )

    ax.set_xlabel("Cosine similarity to the nearest neighbour within the corpus",
                   fontsize=10.5, color="#444444")
    ax.set_ylabel("Number of examples", fontsize=10.5, color="#444444")
    ax.set_title(
        "Calibration of threshold τ: nearest-neighbour similarity distribution "
        f"(n={len(calib_nn):,})",
        fontsize=12.5, color="#333333", pad=14,
    )

    ax.grid(True, axis="y", color="#EDEDED", linewidth=0.6, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#BBBBBB")
    ax.tick_params(colors="#888888", labelsize=9)

    stats_text = (
        f"mean = {mean:.4f}\n"
        f"median = {p50:.4f}\n"
        f"p10 = {p10:.4f}\n"
        f"p25 = {p25:.4f}"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left", color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#DDDDDD", linewidth=0.7))

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")

    return {"mean": mean, "median": p50, "p10": p10, "p25": p25}


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
