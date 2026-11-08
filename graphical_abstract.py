#!/usr/bin/env python3
"""
Graphical abstract (85 x 40 mm): the *same* embedding space of ChEMBL assay
descriptions under two train/val/test splits.

Left  : random split  -> every cluster is present on both sides (near-duplicates leak).
Right : cluster-aware -> whole HDBSCAN clusters go to test, noise goes to train.

Real mode (default) rebuilds the pipeline split from cached embeddings:
  embeddings -> UMAP(50d, cosine) -> HDBSCAN (same parameters as the paper)
  -> cluster_split() logic of train_ner_cluster_aware.py (Random(42) over clusters),
and draws the points on a separate 2-D UMAP projection (visualisation only).
The random split uses the same 80/10/10 ratios.

Usage
-----
  python3 graphical_abstract.py \
      --vecs_cache ~/project/embeddings_a03ce33c.npy \
      --umap_cache ~/project/umap50_cache.npy \
      --out_dir figures --f1_random 0.9196 --f1_cluster 0.8919

  python3 graphical_abstract.py --demo        # synthetic points, layout check only
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ---- palette (custom, warm-neutral + two strong accents) -------------------
INK     = "#1B2733"   # text
TRAIN   = "#9FB3C8"   # pale slate (background mass)
VAL     = "#F2A900"   # gold
TEST    = "#C2185B"   # raspberry
NOISE   = "#DAD3C7"   # pale sand
PANEL   = "#FFFFFF"
FRAME   = "#CFC7B8"
BAD     = "#9C2F3C"   # inflated score chip
GOOD    = "#1F6B5C"   # honest score chip

MM = 1 / 25.4
W, H = 85.0, 40.0
FONT = ["Inter", "Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]


# ---------------------------------------------------------------- data ------
def split_random(n, train_ratio=0.8, val_ratio=0.1, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(n * train_ratio), int(n * val_ratio)
    role = np.empty(n, dtype=int)            # 0 train, 1 val, 2 test
    role[idx[:n_tr]] = 0
    role[idx[n_tr:n_tr + n_va]] = 1
    role[idx[n_tr + n_va:]] = 2
    return role


def split_cluster_aware(labels, train_ratio=0.8, val_ratio=0.1):
    """Same logic as cluster_split() in train_ner_cluster_aware.py."""
    rng = random.Random(42)
    ids = sorted(set(labels.tolist()) - {-1})
    rng.shuffle(ids)
    n_c = len(ids)
    n_tr = max(1, int(n_c * train_ratio))
    n_va = max(1, int(n_c * val_ratio))
    if n_c - n_tr - n_va < 1:
        n_tr -= 1
    if n_va < 1:
        n_va, n_tr = 1, max(1, n_tr - 1)
    tr, va = set(ids[:n_tr]), set(ids[n_tr:n_tr + n_va])
    role = np.full(len(labels), 2, dtype=int)
    for i, c in enumerate(labels):
        if c == -1 or c in tr:
            role[i] = 0
        elif c in va:
            role[i] = 1
    return role


def real_data(args):
    import umap
    if args.labels is not None:
        labels = np.load(args.labels.expanduser()).astype(int)
        vecs = np.load(args.vecs_cache.expanduser()).astype(np.float32)
        assert len(labels) == len(vecs), "labels and embeddings differ in length"
        if args.xy_cache and args.xy_cache.expanduser().exists():
            xy = np.load(args.xy_cache.expanduser())
        else:
            xy = umap.UMAP(n_components=2, n_neighbors=args.nn2d, min_dist=args.md2d, spread=args.spread,
                           metric="cosine", random_state=42, low_memory=True).fit_transform(vecs)
            if args.xy_cache:
                np.save(args.xy_cache.expanduser(), xy)
        print(f"{len(labels)} points, {labels.max() + 1} clusters, noise {np.mean(labels == -1):.1%} (labels from file)")
        return xy, labels
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        from hdbscan import HDBSCAN
    vecs = np.load(args.vecs_cache.expanduser()).astype(np.float32)
    if args.umap_cache and args.umap_cache.expanduser().exists():
        red = np.load(args.umap_cache.expanduser())
    else:
        red = umap.UMAP(n_components=50, n_neighbors=15, min_dist=0.0, metric="cosine",
                        random_state=42, low_memory=True).fit_transform(vecs)
        if args.umap_cache:
            np.save(args.umap_cache.expanduser(), red)
    labels = HDBSCAN(min_cluster_size=30, min_samples=5, cluster_selection_method="eom",
                     metric="euclidean").fit_predict(red).astype(int)
    if args.xy_cache and args.xy_cache.expanduser().exists():
        xy = np.load(args.xy_cache.expanduser())
    else:
        xy = umap.UMAP(n_components=2, n_neighbors=args.nn2d, min_dist=args.md2d, spread=args.spread,
                       metric="cosine", random_state=42, low_memory=True).fit_transform(vecs)
        if args.xy_cache:
            np.save(args.xy_cache.expanduser(), xy)
    print(f"{len(labels)} points, {labels.max() + 1} clusters, "
          f"noise {np.mean(labels == -1):.1%}")
    return xy, labels


def demo_data(seed=3):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.08, 0.92, size=(46, 2))
    pts, lab = [], []
    for k, c in enumerate(centers):
        n = int(rng.integers(30, 110))
        pts.append(c + rng.normal(0, 0.022, size=(n, 2))); lab += [k] * n
    noise = rng.uniform(0.02, 0.98, size=(500, 2))
    pts.append(noise); lab += [-1] * len(noise)
    return np.vstack(pts), np.array(lab)


# ------------------------------------------------------------- drawing ------
def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def draw_panel(ax, xy, role, labels, mark_noise):
    ax.set_facecolor(PANEL)
        # Random draw order, so that overplotting reflects the true 80/10/10 proportions
    # (noise, if marked, is drawn first, underneath everything else).
    jitter = np.random.default_rng(0).random(len(role))
    layer = np.zeros(len(role)) if not mark_noise else np.where(labels == -1, 0.0, 1.0)
    order = np.lexsort((jitter, layer))
    cols = np.array([TRAIN, VAL, TEST])[role]
    if mark_noise:
        cols = np.where(labels == -1, NOISE, cols)
    size = np.array([0.8, 1.5, 1.5])[role]
    alpha = np.array([0.75, 0.95, 0.95])[role]
    if mark_noise:
        alpha = np.where(labels == -1, 0.6, alpha)
    from matplotlib.colors import to_rgba
    rgba = np.array([to_rgba(c, a) for c, a in zip(cols[order], alpha[order])])
    ax.scatter(xy[order, 0], xy[order, 1], s=size[order], c=rgba, lw=0, rasterized=True)
    lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
    pad = 0.03 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(FRAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs_cache", type=Path, default=None)
    ap.add_argument("--umap_cache", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None,
                    help="cluster labels exported by export_cluster_labels.py (recommended)")
    ap.add_argument("--xy_cache", type=Path, default=None, help="cache of the 2-D projection")
    ap.add_argument("--nn2d", type=int, default=12)
    ap.add_argument("--md2d", type=float, default=0.0)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--out_dir", type=Path, default=Path("figures"))
    ap.add_argument("--f1_random", type=float, default=0.9196)
    ap.add_argument("--f1_cluster", type=float, default=0.8919)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        xy, labels = demo_data()
    else:
        if args.vecs_cache is None:
            raise SystemExit("--vecs_cache is required (or use --demo)")
        xy, labels = real_data(args)
    role_rand = split_random(len(labels))
    role_clu = split_cluster_aware(labels)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")

    # ---- grid (mm). Equal gaps: margin 2, gap between panels 11 ------------
    pw, gap, m = 35.0, 11.0, 2.0
    x1, x2 = m, m + pw + gap                         # 2 and 48
    y_panel, h_panel = 11.4, 22.6
    a1 = mm_axes(fig, x1, y_panel, pw, h_panel)
    a2 = mm_axes(fig, x2, y_panel, pw, h_panel)
    draw_panel(a1, xy, role_rand, labels, mark_noise=False)
    draw_panel(a2, xy, role_clu, labels, mark_noise=True)

    texts = []
    def T(x, y, s, **kw):
        t = fig.text(x / W, y / H, s, ha=kw.pop("ha", "center"), va="center", color=kw.pop("color", INK), **kw)
        texts.append(t); return t

    # titles
    T(x1 + pw / 2, 37.0, "Random split", fontsize=7.2, weight="bold")
    T(x2 + pw / 2, 37.0, "Cluster-aware split", fontsize=7.2, weight="bold")

    # legend: stacked in the middle gap, vertically centred on the panels
    gx = x1 + pw + gap / 2
    items = [(TRAIN, "train"), (VAL, "val"), (TEST, "test"), (NOISE, "noise")]
    step = 4.6
    y0 = y_panel + h_panel / 2 + step * (len(items) - 1) / 2
    for i, (c, s) in enumerate(items):
        y = y0 - i * step
        fig.add_artist(plt.Line2D([(gx - 3.9) / W], [y / H], marker="o", ms=3.4, mfc=c, mec="none",
                                  ls="", transform=fig.transFigure))
        T(gx - 2.3, y, s, ha="left", fontsize=6.0)

    # score chips (same width as panels, same height, same baseline)
    ch_y, ch_h = 1.6, 6.6
    def chip(x, color, label, value):
        fig.add_artist(FancyBboxPatch((x / W, ch_y / H), pw / W, ch_h / H,
                       boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
                       fc=color, ec="none"))
        T(x + pw / 2, ch_y + ch_h / 2, f"{label}  {value:.3f}", color="white", fontsize=6.8, weight="bold")
    chip(x1, BAD, "strict F1", args.f1_random)
    chip(x2, GOOD, "strict F1", args.f1_cluster)

    # delta between chips
    d = 100 * (args.f1_random - args.f1_cluster)
    T(gx, ch_y + ch_h / 2, f"−{d:.1f} pp", fontsize=6.4, weight="bold", color=BAD)

    # ---- overlap / overflow check -----------------------------------------
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [(t.get_text(), t.get_window_extent(r)) for t in texts]
    fw, fh = fig.bbox.width, fig.bbox.height
    ok = True
    for i, (s, b) in enumerate(bbs):
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fw or b.y1 > fh:
            print("WARNING text outside canvas:", s); ok = False
        for s2, b2 in bbs[i + 1:]:
            if b.overlaps(b2):
                print("WARNING text overlap:", s, "|", s2); ok = False
    print("layout check:", "OK" if ok else "PROBLEMS")

    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out_dir / f"graphical_abstract.{ext}", dpi=600, facecolor="white")
    print("saved:", args.out_dir / "graphical_abstract.png")


if __name__ == "__main__":
    main()
