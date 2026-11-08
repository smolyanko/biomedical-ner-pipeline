#!/usr/bin/env python3
"""
All figures of the paper / Supplementary Materials from the HDBSCAN pipeline results.

Reads the outputs of the pipeline scripts (all inputs are optional; missing ones are skipped):
  --seed_runs   multi_seed_hdbscan.py      -> seed_runs.csv
  --splits      compare_split_strategies.py -> comparison_full.csv
  --ablations   ablation_train_hdbscan.py  -> directory with *_seeds.csv (or ablation_summary.csv)

Figures (PNG 300 dpi + PDF):
  fig1_main_results        Table 1: strict F1 per architecture, 3 seeds (points + mean ± s.d.)
  fig2_params_vs_f1        trainable parameters (log) vs F1 — parameter efficiency of LoRA
  fig3_split_strategies     F1 by split strategy (random -> cluster-aware), leakage
  fig5_ablations           change in strict F1 relative to the full configuration, per ablation group
  fig6_match_types         strict / partial / exact / ent_type per architecture

Usage
-----
  python3 plot_all_results.py \
      --seed_runs experiments/multi_seed/seed_runs.csv \
      --splits experiments/compare_splits/comparison_full.csv \
      --ablations experiments/ablations_hdbscan \
      --out_dir figures/results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHS = ["bert_linear", "bert_crf", "bert_bilstm_crf", "bert_lora"]
ARCH_LABEL = {"bert_linear": "BERT+Linear", "bert_crf": "BERT+CRF",
              "bert_bilstm_crf": "BERT+BiLSTM+CRF", "bert_lora": "BERT+LoRA+CRF"}
COLORS = {"bert_linear": "#5B7C99", "bert_crf": "#D9A441",
          "bert_bilstm_crf": "#B5475B", "bert_lora": "#3E8E7E"}

# Table 1 (HDBSCAN, 3 seeds) — fallback if seed_runs.csv is not provided
TABLE1_FALLBACK = {"bert_linear": (0.8867, 0.0015), "bert_crf": (0.8919, 0.0050),
                   "bert_bilstm_crf": (0.8959, 0.0020), "bert_lora": (0.8911, 0.0014)}

STRICT = "test_strict_f1"


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })


def save(fig, out_dir: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"[ok] {name}")


def pick_col(df: pd.DataFrame, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


# ───────────────────────── fig1 ─────────────────────────
def fig1_main(seed_df, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs = np.arange(len(ARCHS))
    for i, a in enumerate(ARCHS):
        if seed_df is not None and STRICT in seed_df.columns:
            v = seed_df.loc[seed_df["architecture"] == a, STRICT].values
            m, s = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v,
                       color="k", s=14, zorder=3)
        else:
            m, s = TABLE1_FALLBACK[a]
        ax.bar(i, m, yerr=s, capsize=4, color=COLORS[a], alpha=0.85, width=0.6)
        ax.text(i, m + s + 0.002, f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHS], rotation=12)
    ax.set_ylabel("Strict F1 (test)")
    lo = min(TABLE1_FALLBACK[a][0] if seed_df is None else seed_df.loc[seed_df['architecture'] == a, STRICT].min() for a in ARCHS)
    ax.set_ylim(lo - 0.02, lo + 0.045)
    save(fig, out_dir, "fig1_main_results")


# ───────────────────────── fig2 ─────────────────────────
def fig2_params(seed_df, out_dir):
    if seed_df is None or "n_trainable" not in seed_df.columns or STRICT not in seed_df.columns:
        print("[skip] fig2: seed_runs.csv with n_trainable is required")
        return
    g = seed_df.groupby("architecture").agg(p=("n_trainable", "first"),
                                            f=(STRICT, "mean"), s=(STRICT, "std"))
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for a in ARCHS:
        if a in g.index:
            ax.errorbar(g.loc[a, "p"], g.loc[a, "f"], yerr=g.loc[a, "s"], fmt="o",
                        color=COLORS[a], ms=9, capsize=3, label=ARCH_LABEL[a])
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log)")
    ax.set_ylabel("Strict F1 (test)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out_dir, "fig2_params_vs_f1")


# ───────────────────────── fig3 ─────────────────────────
def fig3_splits(split_df, out_dir):
    if split_df is None or STRICT not in split_df.columns:
        print("[skip] fig3: comparison_full.csv is required")
        return
    strategies = [s for s in ["01_random", "02_stratified_entity",
                              "03_cluster_stratified", "04_cluster_aware"]
                  if s in set(split_df["strategy"])]
    if not strategies:
        strategies = list(dict.fromkeys(split_df["strategy"]))
    pivot = split_df.pivot_table(index="strategy", columns="architecture", values=STRICT)
    archs = [a for a in ["bert_vanilla"] + ARCHS if a in pivot.columns]
    names = {"01_random": "Random", "02_stratified_entity": "Stratified\nby entity type",
             "03_cluster_stratified": "Cluster-stratified\n(leaky control)", "04_cluster_aware": "Cluster-aware"}
    colors = {**COLORS, "bert_vanilla": "#9A9A9A"}
    labels = {**ARCH_LABEL, "bert_vanilla": "BERT+Linear (untuned)"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.8 / len(archs)
    x = np.arange(len(strategies))
    for j, a in enumerate(archs):
        vals = [pivot.loc[s, a] if s in pivot.index else np.nan for s in strategies]
        ax.bar(x + (j - (len(archs) - 1) / 2) * w, vals, w, color=colors[a], label=labels[a])
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(s, s) for s in strategies])
    ax.set_ylabel("Strict F1 (test)")
    lo = np.nanmin(pivot[archs].values) - 0.02
    ax.set_ylim(max(lo, 0), min(np.nanmax(pivot[archs].values) + 0.015, 1))
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, out_dir, "fig3_split_strategies")


# ───────────────────────── fig5 ─────────────────────────
PAIRS = [  # (group, architecture, 'off/control', 'on/full')
    ("Focal loss", "bert_crf", "crf_focal_off", "crf_focal_on"),
    ("Focal loss", "bert_bilstm_crf", "bilstm_focal_off", "bilstm_focal_on"),
    ("BIO→BIOES scheme", "bert_linear", "linear_bio", "linear_bioes"),
    ("BIO→BIOES scheme", "bert_crf", "crf_bio", "crf_bioes"),
    ("Adaptive class weights", "bert_linear", "linear_equal_weights", "linear_adaptive_weights"),
    ("Adaptive class weights", "bert_crf", "crf_equal_weights", "crf_adaptive_weights"),
    ("Layer pooling", "bert_crf", "crf_pooling_off", "crf_pooling_on"),
    ("LoRA warm-start", "bert_lora", "lora_random", "lora_warmstart"),
]


def load_ablations(path: Path):
    if path is None or not path.exists():
        return None
    files = sorted(path.glob("*_seeds.csv"))
    if files:
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = path / "ablation_summary.csv"
    return pd.read_csv(s) if s.exists() else None


def fig5_ablations(abl, out_dir):
    if abl is None:
        print("[skip] fig5: no ablation data")
        return
    col = pick_col(abl, STRICT, "test_strict_f1")
    if col is None or "seed" not in abl.columns:
        print("[skip] fig5: per-seed *_seeds.csv files are required")
        return
    rows = []
    for grp, arch, off, on in PAIRS:
        a = abl.loc[abl["ablation"] == off, col].values
        b = abl.loc[abl["ablation"] == on, col].values
        if len(a) and len(b):
            rows.append((f"{grp}\n{ARCH_LABEL[arch]}", b.mean() - a.mean(),
                         np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                         if len(a) > 1 and len(b) > 1 else 0.0, arch))
    if not rows:
        print("[skip] fig5: no ablation pairs found")
        return
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] * 100 for r in rows], xerr=[r[2] * 100 for r in rows],
            color=[COLORS[r[3]] for r in rows], capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Δ strict F1, pp (full configuration − without component)")
    save(fig, out_dir, "fig5_ablations")


# ───────────────────────── fig6 ─────────────────────────
def fig6_match(seed_df, out_dir):
    if seed_df is None:
        print("[skip] fig6: seed_runs.csv is required")
        return
    cols = [c for c in ["test_strict_f1", "test_partial_f1", "test_exact_f1",
                        "test_ent_type_f1"] if c in seed_df.columns]
    if len(cols) < 2:
        print("[skip] fig6: no nervaluate columns")
        return
    g = seed_df.groupby("architecture")[cols].mean().reindex(ARCHS)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    w = 0.8 / len(cols)
    x = np.arange(len(ARCHS))
    for j, c in enumerate(cols):
        ax.bar(x + (j - (len(cols) - 1) / 2) * w, g[c].values, w,
               label=c.replace("test_", "").replace("_f1", ""))
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHS], rotation=12)
    ax.set_ylabel("F1")
    ax.set_ylim(max(g.min().min() - 0.03, 0), min(g.max().max() + 0.02, 1))
    ax.legend(frameon=False, ncol=len(cols), fontsize=8)
    save(fig, out_dir, "fig6_match_types")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_runs", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--ablations", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=Path("figures/results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()

    seed_df = pd.read_csv(args.seed_runs) if args.seed_runs and args.seed_runs.exists() else None
    split_df = pd.read_csv(args.splits) if args.splits and args.splits.exists() else None
    abl = load_ablations(args.ablations)

    fig1_main(seed_df, args.out_dir)
    fig2_params(seed_df, args.out_dir)
    fig3_splits(split_df, args.out_dir)
    fig5_ablations(abl, args.out_dir)
    fig6_match(seed_df, args.out_dir)
    print(f"Done: {args.out_dir}")


if __name__ == "__main__":
    main()
