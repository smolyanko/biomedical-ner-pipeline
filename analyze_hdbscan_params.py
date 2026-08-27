from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_embeddings(vecs_cache: Path) -> np.ndarray:
    vecs = np.load(str(vecs_cache)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)

def get_umap_reduction(
    vecs: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    cache_path: Path | None,
    seed: int = 42,
) -> np.ndarray:
    if cache_path and cache_path.exists():
        print(f"[umap] Loading cache: {cache_path}")
        reduced = np.load(str(cache_path)).astype(np.float32)
        if len(reduced) != len(vecs):
            raise ValueError(
                f"UMAP cache size mismatch: {len(reduced)} vs {len(vecs)} — "
                "delete cache and recompute"
            )
        return reduced

    import umap as umap_lib
    print(f"[umap] Computing projection (once): {vecs.shape} -> "
          f"({len(vecs)}, {n_components})")
    reducer = umap_lib.UMAP(
        n_components=n_components, n_neighbors=n_neighbors,
        min_dist=min_dist, metric="cosine", random_state=seed,
        low_memory=True, verbose=False,
    )
    reduced = reducer.fit_transform(vecs).astype(np.float32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), reduced)
        print(f"[umap] Saved cache -> {cache_path}")

    return reduced

def run_hdbscan_once(
    reduced: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> dict:
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        backend = "sklearn"
    except ImportError:
        import hdbscan as hdbscan_lib
        backend = "hdbscan_pkg"

    if backend == "sklearn":
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",
            n_jobs=-1,
        )
    else:
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
    labels = clusterer.fit_predict(reduced).astype(int)

    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    n_noise = int(np.sum(labels == -1))
    noise_rate = n_noise / len(labels)

    if n_clusters > 0:
        sizes = np.bincount(labels[labels >= 0])
        median_size = int(np.median(sizes))
        min_size = int(sizes.min())
        max_size = int(sizes.max())
    else:
        median_size = min_size = max_size = 0

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_rate": round(noise_rate, 4),
        "median_cluster_size": median_size,
        "min_cluster_size_actual": min_size,
        "max_cluster_size_actual": max_size,
    }

def run_grid(
    reduced: np.ndarray,
    min_cluster_sizes: List[int],
    min_samples_list: List[int],
    cluster_selection_method: str,
) -> List[dict]:
    rows = []
    combos = list(itertools.product(min_cluster_sizes, min_samples_list))
    for i, (mcs, ms) in enumerate(combos, 1):
        print(f"[grid] {i}/{len(combos)}  min_cluster_size={mcs:>4d}  "
              f"min_samples={ms:>3d}  …", end=" ")
        row = run_hdbscan_once(reduced, mcs, ms, cluster_selection_method)
        print(f"-> {row['n_clusters']} clusters, "
              f"noise {row['noise_rate']:.1%}, "
              f"median_size={row['median_cluster_size']}")
        rows.append(row)
    return rows

def plot_sensitivity(rows: List[dict], out_dir: Path,
                      highlight_mcs: int = 30, highlight_ms: int = 5) -> None:
    min_samples_values = sorted(set(r["min_samples"] for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(min_samples_values)))

    for ms, color in zip(min_samples_values, colors):
        subset = sorted([r for r in rows if r["min_samples"] == ms],
                         key=lambda r: r["min_cluster_size"])
        xs = [r["min_cluster_size"] for r in subset]

        noise_pct = [r["noise_rate"] * 100 for r in subset]
        axes[0].plot(xs, noise_pct, "o-", color=color, label=f"min_samples={ms}")

        n_clust = [r["n_clusters"] for r in subset]
        axes[1].plot(xs, n_clust, "o-", color=color, label=f"min_samples={ms}")

    axes[0].axhspan(20, 25, color="#DDDDDD", alpha=0.4, zorder=0,
                     label="20–25% (reference range)")
    axes[0].axhline(10, color="#E76F51", linestyle=":", linewidth=1,
                     label="10% (sharp drop threshold)")

    current = next((r for r in rows if r["min_cluster_size"] == highlight_mcs
                     and r["min_samples"] == highlight_ms), None)
    if current:
        axes[0].scatter([highlight_mcs], [current["noise_rate"] * 100],
                         color="black", s=90, zorder=5, marker="*",
                         label=f"current choice ({highlight_mcs}, {highlight_ms})")
        axes[1].scatter([highlight_mcs], [current["n_clusters"]],
                         color="black", s=90, zorder=5, marker="*",
                         label=f"current choice ({highlight_mcs}, {highlight_ms})")

    axes[0].set_xlabel("min_cluster_size")
    axes[0].set_ylabel("Noise rate, %")
    axes[0].set_title("HDBSCAN noise rate sensitivity")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("min_cluster_size")
    axes[1].set_ylabel("Number of clusters")
    axes[1].set_title("HDBSCAN cluster count sensitivity")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "hdbscan_sensitivity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")

def print_verdict(rows: List[dict]) -> None:
    noise_rates = [r["noise_rate"] * 100 for r in rows]
    lo, hi = min(noise_rates), max(noise_rates)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Noise rate across parameter grid: {lo:.1f}% – {hi:.1f}%")

    in_band = sum(1 for r in noise_rates if 20 <= r <= 25)
    below_10 = sum(1 for r in noise_rates if r < 10)
    print(f"Combinations in 20-25% range: {in_band} / {len(rows)}")
    print(f"Combinations with noise < 10%:      {below_10} / {len(rows)}")

    if below_10 > 0:
        print(
            "\n-> Combinations with noise < 10% exist — check which "
            "parameters cause this (see table/plot) and "
            "consider revising min_cluster_size/min_samples if these "
            "values are not artifacts of excessive threshold relaxation."
        )
    elif hi - lo < 15:
        print(
            "\n-> Noise rate is relatively stable across parameter variations "
            "(spread < 15 pp.) — current noise level can be described "
            "as a real data property, not a threshold artifact."
        )
    else:
        print(
            "\n-> Noise rate varies noticeably but does not drop below 10% — "
            "picture is ambiguous, visually verify with the plot "
            "and possibly narrow the grid around the current choice for a more accurate "
            "conclusion."
        )
    print("=" * 70)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs-cache", type=Path, required=True)
    ap.add_argument("--umap-cache", type=Path, default=None,
                    help="UMAP projection cache (.npy) — compute once and reuse")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))

    ap.add_argument("--umap-components", type=int, default=50)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)

    ap.add_argument("--min-cluster-sizes", type=int, nargs="+",
                    default=[10, 15, 20, 25, 30, 40, 50, 70, 100])
    ap.add_argument("--min-samples-list", type=int, nargs="+",
                    default=[3, 5, 8, 12])
    ap.add_argument("--cluster-selection-method", default="eom",
                    choices=["eom", "leaf"])

    ap.add_argument("--highlight-mcs", type=int, default=30,
                    help="min_cluster_size of current working choice (for plot marker)")
    ap.add_argument("--highlight-ms", type=int, default=5,
                    help="min_samples of current working choice (for plot marker)")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    vecs = load_embeddings(args.vecs_cache)
    print(f"[data] {len(vecs):,} vectors, dim={vecs.shape[1]}")

    reduced = get_umap_reduction(
        vecs,
        n_components=args.umap_components,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        cache_path=args.umap_cache,
        seed=args.seed,
    )

    rows = run_grid(
        reduced,
        min_cluster_sizes=args.min_cluster_sizes,
        min_samples_list=args.min_samples_list,
        cluster_selection_method=args.cluster_selection_method,
    )

    with (args.out_dir / "hdbscan_sensitivity.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"[save] {args.out_dir / 'hdbscan_sensitivity.json'}")

    import csv
    with (args.out_dir / "hdbscan_sensitivity.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {args.out_dir / 'hdbscan_sensitivity.csv'}")

    plot_sensitivity(rows, args.out_dir,
                      highlight_mcs=args.highlight_mcs,
                      highlight_ms=args.highlight_ms)

    print_verdict(rows)

if __name__ == "__main__":
    main()
