from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

PASTEL_PALETTE = [
    "#A8D8EA",
    "#F6D6AD",
    "#B8E0D2",
    "#F7A9A8",
    "#D6CDEA",
    "#FFE8A3",
    "#B5EAD7",
    "#FFB7B2",
    "#C7CEEA",
    "#FFDAC1",
]
NOISE_COLOR = "#D9D9D9"
NOISE_ALPHA = 0.35
CLUSTER_ALPHA = 0.75
BACKGROUND = "#FFFFFF"
GRID_COLOR = "#EDEDED"

def color_for_cluster(cid: int, n_clusters: int) -> str:
    if cid < 0:
        return NOISE_COLOR
    return PASTEL_PALETTE[cid % len(PASTEL_PALETTE)]

def load_embeddings(vecs_cache: Path) -> np.ndarray:
    vecs = np.load(str(vecs_cache)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    return vecs

def run_hdbscan(
    vecs: np.ndarray,
    umap_components: int,
    umap_neighbors: int,
    umap_min_dist: float,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    seed: int = 42,
) -> np.ndarray:
    import umap as umap_lib
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        backend = "sklearn"
    except ImportError:
        import hdbscan as hdbscan_lib
        backend = "hdbscan_pkg"

    print(f"[cluster] UMAP for clustering: {vecs.shape} -> "
          f"({len(vecs)}, {umap_components})")
    reducer = umap_lib.UMAP(
        n_components=umap_components,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=seed,
        low_memory=True,
        verbose=False,
    )
    reduced = reducer.fit_transform(vecs).astype(np.float32)

    print(f"[cluster] HDBSCAN (backend={backend}): "
          f"min_cluster_size={min_cluster_size} min_samples={min_samples}")
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
    print(f"[cluster] found {n_clusters} clusters, "
          f"{n_noise} noise points ({n_noise / len(labels):.1%})")

    return labels

def project_for_plot(
    vecs: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seed: int = 42,
) -> np.ndarray:
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        low_memory=True,
        verbose=False,
    )
    return reducer.fit_transform(vecs).astype(np.float32)

def build_legend_handles(labels: np.ndarray, top_n: int = 9):
    from collections import Counter
    sizes = Counter(labels[labels >= 0].tolist())
    top_clusters = [cid for cid, _ in sizes.most_common(top_n)]

    handles = []
    for rank, cid in enumerate(top_clusters):
        handles.append(Patch(
            facecolor=PASTEL_PALETTE[rank % len(PASTEL_PALETTE)],
            edgecolor="none",
            label=f"cluster {cid} (n={sizes[cid]})",
        ))
    n_other = sum(v for k, v in sizes.items() if k not in top_clusters)
    if n_other > 0:
        handles.append(Patch(facecolor="#BBBBBB", edgecolor="none",
                              label=f"other clusters (n={n_other})"))
    n_noise = int(np.sum(labels == -1))
    if n_noise > 0:
        handles.append(Patch(facecolor=NOISE_COLOR, edgecolor="none",
                              label=f"noise (n={n_noise})"))
    return handles, top_clusters

def colors_for_plot(labels: np.ndarray, top_clusters: list) -> np.ndarray:
    colors = np.empty(len(labels), dtype=object)
    for i, cid in enumerate(labels):
        if cid == -1:
            colors[i] = NOISE_COLOR
        elif cid in top_clusters:
            rank = top_clusters.index(cid)
            colors[i] = PASTEL_PALETTE[rank % len(PASTEL_PALETTE)]
        else:
            colors[i] = "#BBBBBB"
    return colors

def alphas_for_plot(labels: np.ndarray) -> np.ndarray:
    return np.where(labels == -1, NOISE_ALPHA, CLUSTER_ALPHA)

def plot_2d(coords: np.ndarray, labels: np.ndarray, out_path: Path,
            title: str, top_n: int = 9) -> None:
    handles, top_clusters = build_legend_handles(labels, top_n=top_n)
    colors = colors_for_plot(labels, top_clusters)
    alphas = alphas_for_plot(labels)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    noise_mask = labels == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
               c=NOISE_COLOR, s=10, alpha=NOISE_ALPHA,
               linewidths=0, label=None, zorder=1)
    ax.scatter(coords[~noise_mask, 0], coords[~noise_mask, 1],
               c=list(colors[~noise_mask]), s=14, alpha=CLUSTER_ALPHA,
               linewidths=0, zorder=2)

    ax.set_xlabel("UMAP-1", fontsize=11, color="#555555")
    ax.set_ylabel("UMAP-2", fontsize=11, color="#555555")
    ax.set_title(title, fontsize=14, color="#333333", pad=14)

    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#888888", labelsize=9)

    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
               frameon=False, fontsize=9, title="Clusters (top by size)",
               title_fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")

def plot_3d(coords: np.ndarray, labels: np.ndarray, out_path: Path,
            title: str, top_n: int = 9, elev: float = 20, azim: float = -60) -> None:
    handles, top_clusters = build_legend_handles(labels, top_n=top_n)
    colors = colors_for_plot(labels, top_clusters)

    fig = plt.figure(figsize=(11, 9), facecolor=BACKGROUND)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BACKGROUND)

    noise_mask = labels == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1], coords[noise_mask, 2],
               c=NOISE_COLOR, s=8, alpha=NOISE_ALPHA, linewidths=0, zorder=1)
    ax.scatter(coords[~noise_mask, 0], coords[~noise_mask, 1], coords[~noise_mask, 2],
               c=list(colors[~noise_mask]), s=12, alpha=CLUSTER_ALPHA,
               linewidths=0, zorder=2)

    ax.set_xlabel("UMAP-1", fontsize=10, color="#555555", labelpad=8)
    ax.set_ylabel("UMAP-2", fontsize=10, color="#555555", labelpad=8)
    ax.set_zlabel("UMAP-3", fontsize=10, color="#555555", labelpad=8)
    ax.set_title(title, fontsize=14, color="#333333", pad=6)

    ax.xaxis.pane.set_facecolor(BACKGROUND)
    ax.yaxis.pane.set_facecolor(BACKGROUND)
    ax.zaxis.pane.set_facecolor(BACKGROUND)
    ax.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors="#888888", labelsize=8)
    ax.view_init(elev=elev, azim=azim)

    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5),
               frameon=False, fontsize=9, title="Clusters (top by size)",
               title_fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")

def plot_3d_interactive(coords: np.ndarray, labels: np.ndarray, out_path: Path,
                         title: str, top_n: int = 9) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[plot] plotly not installed — skipping interactive 3D "
              "(pip install plotly if .html is needed)")
        return

    handles, top_clusters = build_legend_handles(labels, top_n=top_n)
    colors = colors_for_plot(labels, top_clusters)

    fig = go.Figure()
    noise_mask = labels == -1
    fig.add_trace(go.Scatter3d(
        x=coords[noise_mask, 0], y=coords[noise_mask, 1], z=coords[noise_mask, 2],
        mode="markers",
        marker=dict(size=2.5, color=NOISE_COLOR, opacity=NOISE_ALPHA),
        name="noise",
    ))
    for rank, cid in enumerate(top_clusters):
        m = labels == cid
        fig.add_trace(go.Scatter3d(
            x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
            mode="markers",
            marker=dict(size=3, color=PASTEL_PALETTE[rank % len(PASTEL_PALETTE)],
                        opacity=CLUSTER_ALPHA),
            name=f"cluster {cid} (n={int(m.sum())})",
        ))
    other_mask = (~noise_mask) & (~np.isin(labels, top_clusters))
    if other_mask.any():
        fig.add_trace(go.Scatter3d(
            x=coords[other_mask, 0], y=coords[other_mask, 1], z=coords[other_mask, 2],
            mode="markers",
            marker=dict(size=2.5, color="#BBBBBB", opacity=CLUSTER_ALPHA),
            name="other clusters",
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3",
            bgcolor=BACKGROUND,
        ),
        paper_bgcolor=BACKGROUND,
        legend_title_text="Clusters",
        font=dict(size=11, color="#333333"),
    )
    fig.write_html(str(out_path))
    print(f"[plot] Saved (interactive): {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs-cache", type=Path, required=True)
    ap.add_argument("--labels-cache", type=Path, default=None,
                    help="Precomputed HDBSCAN labels (.npy) — skips recomputation")
    ap.add_argument("--save-labels", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))

    ap.add_argument("--umap-components", type=int, default=50)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--min-cluster-size", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"])

    ap.add_argument("--plot-neighbors", type=int, default=15)
    ap.add_argument("--plot-min-dist", type=float, default=0.3,
                    help="Larger than for clustering — for readable plot")

    ap.add_argument("--top-n-legend", type=int, default=9)
    ap.add_argument("--interactive", action="store_true",
                    help="Also save interactive 3D (.html, requires plotly)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vecs = load_embeddings(args.vecs_cache)
    print(f"[data] {len(vecs):,} vectors, dim={vecs.shape[1]}")

    if args.labels_cache and args.labels_cache.exists():
        print(f"[cluster] Loading saved labels: {args.labels_cache}")
        labels = np.load(str(args.labels_cache)).astype(int)
        assert len(labels) == len(vecs), "labels/vecs size mismatch"
    else:
        labels = run_hdbscan(
            vecs,
            umap_components=args.umap_components,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_method=args.cluster_selection_method,
            seed=args.seed,
        )
        if args.save_labels:
            np.save(str(args.save_labels), labels)
            print(f"[cluster] Saved labels -> {args.save_labels}")

    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    n_noise = int(np.sum(labels == -1))

    print("[plot] Building 2D projection (separate UMAP, larger min_dist)…")
    coords_2d = project_for_plot(
        vecs, n_components=2,
        n_neighbors=args.plot_neighbors, min_dist=args.plot_min_dist,
        seed=args.seed,
    )
    plot_2d(
        coords_2d, labels, args.out_dir / "embeddings_2d_hdbscan.png",
        title=f"Train corpus (n={len(vecs):,}): HDBSCAN clusters, 2D UMAP\n"
              f"{n_clusters} clusters, noise {n_noise:,} ({n_noise/len(vecs):.1%})",
        top_n=args.top_n_legend,
    )

    print("[plot] Building 3D projection (separate UMAP, larger min_dist)…")
    coords_3d = project_for_plot(
        vecs, n_components=3,
        n_neighbors=args.plot_neighbors, min_dist=args.plot_min_dist,
        seed=args.seed,
    )
    plot_3d(
        coords_3d, labels, args.out_dir / "embeddings_3d_hdbscan.png",
        title=f"Train corpus (n={len(vecs):,}): HDBSCAN clusters, 3D UMAP",
        top_n=args.top_n_legend,
    )

    if args.interactive:
        plot_3d_interactive(
            coords_3d, labels, args.out_dir / "embeddings_3d_hdbscan.html",
            title=f"Train corpus: HDBSCAN clusters, 3D UMAP (interactive)",
            top_n=args.top_n_legend,
        )

    summary = {
        "n_samples": len(vecs),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_rate": round(n_noise / len(vecs), 4),
    }
    with (args.out_dir / "hdbscan_plot_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
