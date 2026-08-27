from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np

NOISE_COLOR = "#E0E0E0"

def generate_distinct_colors(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = rng.random()
    for i in range(n):
        h = (h + golden_ratio_conjugate) % 1.0
        s = 0.55 + 0.35 * ((i * 7) % 5) / 4
        v = 0.75 + 0.20 * ((i * 3) % 4) / 3
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors

def load_embeddings(vecs_cache: Path) -> np.ndarray:
    vecs = np.load(str(vecs_cache)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)

def run_hdbscan(vecs, umap_components, umap_neighbors, umap_min_dist,
                 min_cluster_size, min_samples, cluster_selection_method, seed=42):
    import umap as umap_lib
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        backend = "sklearn"
    except ImportError:
        import hdbscan as hdbscan_lib
        backend = "hdbscan_pkg"

    reducer = umap_lib.UMAP(
        n_components=umap_components, n_neighbors=umap_neighbors,
        min_dist=umap_min_dist, metric="cosine", random_state=seed,
        low_memory=True, verbose=False,
    )
    reduced = reducer.fit_transform(vecs).astype(np.float32)

    if backend == "sklearn":
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean", n_jobs=-1,
        )
    else:
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean", core_dist_n_jobs=-1,
        )
    labels = clusterer.fit_predict(reduced).astype(int)
    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    print(f"[cluster] {n_clusters} clusters, {int((labels == -1).sum())} noise")
    return labels

def project_3d(vecs, n_neighbors, min_dist, seed=42):
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=seed, low_memory=True, verbose=False,
    )
    return reducer.fit_transform(vecs).astype(np.float32)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs-cache", type=Path, required=True)
    ap.add_argument("--labels-cache", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--umap-components", type=int, default=50)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--min-cluster-size", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--cluster-selection-method", default="eom")
    ap.add_argument("--plot-neighbors", type=int, default=15)
    ap.add_argument("--plot-min-dist", type=float, default=0.3)
    ap.add_argument("--point-size", type=float, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import plotly.graph_objects as go
    except ImportError:
        raise SystemExit("plotly required: pip install plotly")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    vecs = load_embeddings(args.vecs_cache)
    print(f"[data] {len(vecs):,} vectors, dim={vecs.shape[1]}")

    if args.labels_cache and args.labels_cache.exists():
        labels = np.load(str(args.labels_cache)).astype(int)
    else:
        labels = run_hdbscan(
            vecs, args.umap_components, args.umap_neighbors, args.umap_min_dist,
            args.min_cluster_size, args.min_samples, args.cluster_selection_method,
            seed=args.seed,
        )

    coords = project_3d(vecs, args.plot_neighbors, args.plot_min_dist, seed=args.seed)

    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 1
    palette = generate_distinct_colors(n_clusters, seed=args.seed)
    point_colors = [NOISE_COLOR if cid == -1 else palette[cid] for cid in labels]

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=args.point_size,
            color=point_colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        hoverinfo="skip",
        showlegend=False,
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    out_path = args.out_dir / "embeddings_3d_all_clusters_interactive.html"
    fig.write_html(str(out_path))
    print(f"[plot] Saved: {out_path}")

if __name__ == "__main__":
    main()
