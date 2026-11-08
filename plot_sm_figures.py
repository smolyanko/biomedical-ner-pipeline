#!/usr/bin/env python3
"""
Supplementary-material figures that are not produced by the training scripts.
Figures carry no titles (captions belong in the Supplementary text); axis labels
only state what is plotted.

  corpus_size      Corpus growth over the iterative expansion stages
  token_length     Token-length distribution of the annotated corpus  -> Table S1 numbers too
  split_composition  Cluster-size distribution and train/val/test composition -> Table S2 numbers
  learning_curves  Training loss and dev F1 per epoch as a density of overlapping runs, parsed from logs
  per_type_metrics Strict precision / recall / F1 for SUB and TAR (from taxonomy_summary.csv)

Usage
-----
  python3 plot_sm_figures.py corpus_size --stage_sizes 1200 15000 16567
  python3 plot_sm_figures.py token_length --data_path train.json
  python3 plot_sm_figures.py split_composition --labels cluster_labels.npy
  python3 plot_sm_figures.py learning_curves --logs 'experiments/run1/bert_bilstm_crf/*.log' --final_logs 'experiments/multi_seed/*.log' --arch bert_bilstm_crf
  python3 plot_sm_figures.py per_type_metrics --taxonomy experiments/error_analysis/taxonomy_summary.csv
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import matplotlib.colors  # noqa: F401
import numpy as np
import pandas as pd

INK = "#1B2733"
BLUE, GOLD, RASP, TEAL, GREY = "#5B7C99", "#D9A441", "#B5475B", "#3E8E7E", "#B9B2A6"
FONT = ["Inter", "Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 9,
        "axes.edgecolor": "#888888", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "savefig.bbox": "tight", "figure.facecolor": "white",
    })


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"[ok] {out_dir / name}.png")


# ------------------------------------------------------------ corpus size ----
def cmd_corpus_size(a):
    sizes = a.stage_sizes
    labels = (a.stage_labels if a.stage_labels else
              ["Manual seed"] + [f"Expansion stage {i}" for i in range(1, len(sizes))])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bars = ax.bar(range(len(sizes)), sizes, color=[GOLD] + [BLUE] * (len(sizes) - 1), width=0.6)
    for b, v in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Annotated examples")
    ax.set_ylim(0, max(sizes) * 1.12)
    ax.grid(axis="x", visible=False)
    save(fig, a.out_dir, "figS_corpus_size")


# ----------------------------------------------------------- token length ----
def length_stats(lengths):
    x = np.asarray(lengths)
    return {"n": len(x), "min": int(x.min()), "max": int(x.max()), "mean": float(x.mean()),
            "median": float(np.median(x)), "sd": float(x.std(ddof=1)),
            "P90": float(np.percentile(x, 90)), "P95": float(np.percentile(x, 95)),
            "P99": float(np.percentile(x, 99))}


def cmd_token_length(a):
    import train_ner_cluster_aware as tnc
    ex = tnc.clean_examples(tnc.load_json(Path(a.data_path).expanduser()))
    texts = [tnc._safe_text(e) for e in ex]
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        lengths = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
        unit = "Tokens per description (subword)"
    else:
        lengths = [len(t.split()) for t in texts]
        unit = "Words per description"
    st = length_stats(lengths)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([st]).to_csv(a.out_dir / "token_length_stats.csv", index=False)
    print(st)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    hi = float(np.percentile(lengths, 99.5))
    ax.hist(np.clip(lengths, 0, hi), bins=60, color=BLUE, edgecolor="white", lw=0.3)
    ax.axvline(st["median"], color=RASP, lw=1.2)
    ax.text(st["median"], ax.get_ylim()[1] * 0.97, f" median {st['median']:.0f}", color=RASP,
            va="top", fontsize=8)
    ax.set_xlabel(unit)
    ax.set_ylabel("Examples")
    save(fig, a.out_dir, "figS_token_length")


# ------------------------------------------------------ split composition ----
def composition(clusters, roles):
    """clusters: array of cluster ids (-1 = noise); roles: array of 'train'/'val'/'test'."""
    clusters, roles = np.asarray(clusters), np.asarray(roles)
    sizes = Counter(clusters[clusters >= 0].tolist())
    s = np.array(list(sizes.values()))
    rows = {"n_examples": len(clusters), "n_clusters": len(sizes),
            "noise_n": int((clusters < 0).sum()), "noise_pct": float(100 * (clusters < 0).mean()),
            "cluster_size_min": int(s.min()), "cluster_size_median": float(np.median(s)),
            "cluster_size_max": int(s.max())}
    for r in ("train", "val", "test"):
        m = roles == r
        rows[f"{r}_examples"] = int(m.sum())
        rows[f"{r}_clusters"] = int(len(set(clusters[m & (clusters >= 0)].tolist())))
    return rows, s


def roles_from_labels(labels, train_ratio=0.8, val_ratio=0.1):
    """Same logic as cluster_split() in train_ner_cluster_aware.py."""
    import random
    rng = random.Random(42)
    ids = sorted(set(labels.tolist()) - {-1})
    rng.shuffle(ids)
    n_c = len(ids)
    n_tr, n_va = max(1, int(n_c * train_ratio)), max(1, int(n_c * val_ratio))
    if n_c - n_tr - n_va < 1:
        n_tr -= 1
    tr, va = set(ids[:n_tr]), set(ids[n_tr:n_tr + n_va])
    return np.array(["train" if (c == -1 or c in tr) else "val" if c in va else "test" for c in labels])


def cmd_split_composition(a):
    if a.labels is not None:
        cl = np.load(a.labels.expanduser()).astype(int)
        ro = roles_from_labels(cl)
    else:
        import logging
        import train_ner_cluster_aware as tnc
        log = logging.getLogger("split"); log.addHandler(logging.StreamHandler()); log.setLevel(logging.INFO)
        tr, va, te = tnc.load_and_split(
            data_path=Path(a.data_path).expanduser(), train_ratio=0.8, val_ratio=0.1, logger=log,
            vecs_cache=a.vecs_cache, umap_cache=a.umap_cache,
            min_cluster_size=a.min_cluster_size, min_samples=a.min_samples)
        cl, ro = [], []
        for name, part in (("train", tr), ("val", va), ("test", te)):
            for e in part:
                c = e["meta"]["cluster"]
                cl.append(-1 if c == tnc.NOISE_CLUSTER_ID else c); ro.append(name)
    rows, sizes = composition(cl, ro)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rows]).to_csv(a.out_dir / "split_composition.csv", index=False)
    print(rows)

    cl, ro = np.array(cl), np.array(ro)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.hist(sizes, bins=np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 28),
             color=BLUE, edgecolor="white", lw=0.3)
    ax1.set_xscale("log")
    ax1.set_xlabel("Cluster size (examples)")
    ax1.set_ylabel("Clusters")
    ax1.text(-0.2, 1.04, "(a)", transform=ax1.transAxes, fontweight="bold")
    parts = [("Train", (ro == "train") & (cl >= 0), BLUE), ("Train (noise)", (ro == "train") & (cl < 0), GREY),
             ("Validation", ro == "val", GOLD), ("Test", ro == "test", RASP)]
    left = 0
    for i, (lab, m, col) in enumerate(parts):
        n = int(m.sum())
        ax2.barh(0, n, left=left, color=col, edgecolor="white", lw=0.5, label=f"{lab} ({n:,})")
        left += n
    ax2.set_yticks([]); ax2.set_xlabel("Examples"); ax2.grid(axis="y", visible=False)
    ax2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
    ax2.text(-0.12, 1.04, "(b)", transform=ax2.transAxes, fontweight="bold")
    save(fig, a.out_dir, "figS_split_composition")


# --------------------------------------------------------- learning curves ----
EPOCH_RE = re.compile(r"\[(\w+)\]\s+epoch=(\d+)/(\d+)\s+loss=([\d.]+)\s+dev_f1=([\d.]+)")
SEED_RE = re.compile(r"^\s*(bert_\w+)\s+seed=(\d+)\s*$")


def parse_logs(paths):
    """Return DataFrame(arch, seed, run, epoch, loss, dev_f1). A new run starts when
    the epoch counter restarts or a '<arch>  seed=N' banner is met."""
    rows, run, seed, prev = [], 0, None, 0
    for p in paths:
        for line in Path(p).read_text(errors="ignore").splitlines():
            m = SEED_RE.match(line)
            if m:
                seed = int(m.group(2))
            m = EPOCH_RE.search(line)
            if not m:
                continue
            arch, ep = m.group(1), int(m.group(2))
            if ep <= prev:
                run += 1
            prev = ep
            rows.append({"arch": arch, "seed": seed, "run": run, "epoch": ep,
                         "loss": float(m.group(4)), "dev_f1": float(m.group(5))})
    return pd.DataFrame(rows)


def make_cmap(spec):
    """spec: a matplotlib colormap name, or comma-separated hex colours (light -> dark)."""
    from matplotlib.colors import LinearSegmentedColormap
    if "#" in spec:
        return LinearSegmentedColormap.from_list("custom", [c.strip() for c in spec.split(",")])
    return plt.get_cmap(spec)


DEFAULT_CMAP = "#FFFFFF,#DCE5EC,#9FB3C8,#5B7C99,#3E8E7E,#D9A441,#B5475B"


def density_grid(runs, col, nx, ny, ylim, oversample=24):
    """Rasterise all trajectories into an (ny, nx) grid; each cell counts how many
    trajectories cross it (the more overlapping runs, the higher the count)."""
    H = np.zeros((ny, nx))
    xmax = max(g["epoch"].max() for g in runs)
    for g in runs:
        x, y = g["epoch"].values.astype(float), g[col].values
        if len(x) < 2:
            continue
        xs = np.linspace(x.min(), x.max(), oversample * (len(x) - 1) + 1)
        ys = np.interp(xs, x, y)
        xi = np.clip(((xs - 0.5) / xmax * nx).astype(int), 0, nx - 1)
        yi = np.clip(((ys - ylim[0]) / (ylim[1] - ylim[0]) * ny).astype(int), 0, ny - 1)
        cells = set(zip(yi.tolist(), xi.tolist()))   # a run counts once per cell
        for r, c in cells:
            H[r, c] += 1
    return H, xmax


def cmd_learning_curves(a):
    import glob
    expand = lambda pats: [p for pat in pats for p in sorted(glob.glob(str(Path(pat).expanduser())))]
    df = parse_logs(expand(a.logs))
    df = df[df["arch"] == a.arch]
    if df.empty:
        raise SystemExit(f"No epoch lines found for {a.arch}")
    runs = [g.sort_values("epoch") for _, g in df.groupby("run")]
    print(f"{len(runs)} training runs of {a.arch}")
    cmap = make_cmap(a.cmap)
    final = []
    if a.final_logs:
        fd = parse_logs(expand(a.final_logs))
        fd = fd[fd["arch"] == a.arch]
        final = [g.sort_values("epoch") for _, g in fd.groupby("run")]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    for ax, col, lab, tag in ((axes[0], "loss", "Training loss", "(a)"),
                              (axes[1], "dev_f1", "Validation F1 (seqeval, strict)", "(b)")):
        allv = np.concatenate([g[col].values for g in runs])
        ylim = (np.percentile(allv, 0.5), np.percentile(allv, 99.5))
        pad = 0.04 * (ylim[1] - ylim[0]); ylim = (ylim[0] - pad, ylim[1] + pad)
        for g in final:
            ylim = (min(ylim[0], g[col].min()), max(ylim[1], g[col].max()))
        H, xmax = density_grid(runs, col, a.nx, a.ny, ylim)
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=a.smooth) / len(runs) * 100        # % of runs
        Hn = np.where(H > 0.02 * H.max(), H, np.nan)
        norm = matplotlib.colors.PowerNorm(gamma=0.6, vmin=0, vmax=H.max())
        im = ax.imshow(Hn, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                       extent=[0.5, xmax + 0.5, ylim[0], ylim[1]], interpolation="nearest")
        for g in final:
            ax.plot(g["epoch"], g[col], color=INK, lw=0.9, alpha=0.9)
        ax.set_xlim(0.5, xmax + 0.5); ax.set_ylim(*ylim)
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab); ax.grid(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.text(-0.22, 1.04, tag, transform=ax.transAxes, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label("Share of runs, %", fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, a.out_dir, "figS_learning_curves")


# ---------------------------------------------------- per-type P / R / F1 ----
def per_type_prf(tax: pd.DataFrame) -> pd.DataFrame:
    """Strict per-type precision/recall/F1 from the span taxonomy (mean over seeds).
    gold side of type T: TP, boundary_* (predicted as T with wrong boundary), missed,
    type_confusion / boundary_and_type (predicted as the other type).
    predicted spans of type T = TP + boundary_*(T gold) + confusions from the other
    type's gold + spurious(T) + fragment(T)."""
    g = tax[tax["role"] == "gold"].pivot_table(index="type", columns="kind", values="mean", aggfunc="first").fillna(0)
    p = tax[tax["role"] == "pred"].pivot_table(index="type", columns="kind", values="mean", aggfunc="first").fillna(0)
    types = list(g.index)
    bnd = [c for c in g.columns if c.startswith("boundary_") and c != "boundary_and_type"]
    conf = [c for c in ("type_confusion", "boundary_and_type") if c in g.columns]
    rows = []
    for t in types:
        gold_n = g.loc[t].sum()
        pred_n = (g.loc[t, "TP"] + g.loc[t, bnd].sum()
                  + sum(g.loc[o, conf].sum() for o in types if o != t)
                  + (p.loc[t].sum() if t in p.index else 0))
        pr, rc = g.loc[t, "TP"] / pred_n, g.loc[t, "TP"] / gold_n
        rows.append({"type": t, "gold_spans": gold_n, "pred_spans": pred_n, "precision": pr,
                     "recall": rc, "f1": 2 * pr * rc / (pr + rc)})
    return pd.DataFrame(rows)


def cmd_per_type_metrics(a):
    df = per_type_prf(pd.read_csv(a.taxonomy))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_dir / "per_type_metrics.csv", index=False)
    print(df.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    w = 0.26
    for j, (m, col) in enumerate((("precision", BLUE), ("recall", GOLD), ("f1", RASP))):
        v = df[m].values
        bars = ax.bar(np.arange(len(df)) + (j - 1) * w, v, w, color=col, label=m.capitalize() if m != "f1" else "F1")
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x + 0.004, f"{x:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["type"])
    ax.set_ylabel("Strict score (test)")
    ax.set_ylim(max(df[["precision", "recall", "f1"]].min().min() - 0.08, 0), 1.0)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="x", visible=False)
    save(fig, a.out_dir, "figS_per_type_metrics")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        p.add_argument("--out_dir", type=Path, default=Path("figures/supplementary"))
        return p

    p = add("corpus_size", cmd_corpus_size)
    p.add_argument("--stage_sizes", type=int, nargs="+", required=True)
    p.add_argument("--stage_labels", nargs="+", default=None)
    p = add("token_length", cmd_token_length)
    p.add_argument("--data_path", required=True)
    p.add_argument("--tokenizer", default=None, help="HF tokenizer name; omit to count words")
    p = add("split_composition", cmd_split_composition)
    p.add_argument("--data_path", default=None)
    p.add_argument("--labels", type=Path, default=None, help="labels from export_cluster_labels.py")
    p.add_argument("--vecs_cache", type=Path, default=None)
    p.add_argument("--umap_cache", type=Path, default=None)
    p.add_argument("--min_cluster_size", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=5)
    p = add("learning_curves", cmd_learning_curves)
    p.add_argument("--logs", nargs="+", required=True,
                   help="training logs (files or globs); every run found for --arch is drawn, "
                        "e.g. all Optuna trials plus the final runs")
    p.add_argument("--final_logs", nargs="*", default=None,
                   help="optional logs of the final per-seed runs, drawn as thin lines on top")
    p.add_argument("--arch", default="bert_bilstm_crf")
    p.add_argument("--cmap", default=DEFAULT_CMAP,
                   help="matplotlib colormap name or comma-separated hex colours, light to dark")
    p.add_argument("--nx", type=int, default=150)
    p.add_argument("--ny", type=int, default=120)
    p.add_argument("--smooth", type=float, default=1.6, help="gaussian smoothing of the density, in cells")
    p = add("per_type_metrics", cmd_per_type_metrics)
    p.add_argument("--taxonomy", type=Path, required=True)

    a = ap.parse_args()
    style()
    a.fn(a)


if __name__ == "__main__":
    main()
