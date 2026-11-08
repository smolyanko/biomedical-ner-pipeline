#!/usr/bin/env python3
"""
Near-duplicate leakage check: how similar are test examples to the training set?

For each split strategy, every test example is compared (cosine similarity of the cached
embeddings) with all training examples; we report the distribution of the maximum similarity.
  * cluster-aware: reproduces the pipeline split from cluster_labels.npy (noise -> train)
  * random: 80/10/10 random splits with the same sizes (several seeds; mean +- s.d.)
Also reports the share of descriptions that occur more than once in the corpus (exact text)
and the share of test descriptions that also occur verbatim in train.

Outputs (out_dir): leakage_nn_stats.csv, leakage_text_duplicates.json,
                   figS_leakage_nn.png/.pdf

Usage (from the repository root):
  python3 leakage_nn_check.py --data_path DATA.json --vecs_cache EMB.npy \
      --labels cluster_labels.npy --out_dir figures/supplementary
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_sm_figures as P

THRESHOLDS = (0.99, 0.95, 0.90, 0.8429)


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def max_sim(V, test_idx, train_idx, chunk=512):
    """max cosine similarity of each test vector to any train vector (V unit-normalised)."""
    out = np.empty(len(test_idx), dtype=np.float32)
    T = V[train_idx]
    for i in range(0, len(test_idx), chunk):
        out[i:i + chunk] = (V[test_idx[i:i + chunk]] @ T.T).max(axis=1)
    return out


def summarise(name, seed, ms, txt_dup):
    row = {"split": name, "seed": seed, "n_test": len(ms), "median_max_sim": float(np.median(ms)),
           "mean_max_sim": float(ms.mean()), "test_text_in_train_pct": 100 * txt_dup}
    for t in THRESHOLDS:
        row[f"share_ge_{t}"] = float((ms >= t).mean())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=Path, required=True)
    ap.add_argument("--vecs_cache", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("figures/supplementary"))
    ap.add_argument("--n_random", type=int, default=5, help="number of random splits")
    a = ap.parse_args()

    import train_ner_cluster_aware as tnc
    ex = tnc.clean_examples(tnc.load_json(a.data_path.expanduser()))
    texts = [norm_text(tnc._safe_text(e)) for e in ex]
    V = np.load(a.vecs_cache.expanduser()).astype(np.float32)
    labels = np.load(a.labels.expanduser()).astype(int)
    assert len(texts) == len(V) == len(labels), (len(texts), len(V), len(labels))
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)   # cosine
    n = len(V)

    # --- exact-text duplicates in the whole corpus
    cnt = pd.Series(texts).value_counts()
    dup_stats = {"n_examples": n, "n_unique_texts": int(len(cnt)),
                 "share_examples_with_repeated_text": float((pd.Series(texts).map(cnt) > 1).mean()),
                 "largest_repeat": int(cnt.iloc[0])}

    def text_in_train(test_idx, train_idx):
        tr = {texts[i] for i in train_idx}
        return float(np.mean([texts[i] in tr for i in test_idx]))

    rows, curves = [], {}
    # --- cluster-aware (pipeline split)
    roles = P.roles_from_labels(labels)
    tr, te = np.where(roles == "train")[0], np.where(roles == "test")[0]
    ms = max_sim(V, te, tr)
    rows.append(summarise("cluster-aware", 0, ms, text_in_train(te, tr)))
    curves["Cluster-aware"] = [ms]
    # --- random splits of identical sizes
    n_te = len(te); n_va = int((roles == "val").sum())
    rs = []
    for seed in range(args_n := a.n_random):
        perm = np.random.RandomState(seed).permutation(n)
        te_r, tr_r = perm[:n_te], perm[n_te + n_va:]
        m = max_sim(V, te_r, tr_r)
        rows.append(summarise("random", seed, m, text_in_train(te_r, tr_r)))
        rs.append(m)
    curves["Random"] = rs

    df = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_dir / "leakage_nn_stats.csv", index=False)
    (a.out_dir / "leakage_text_duplicates.json").write_text(json.dumps(dup_stats, indent=2))
    print(df.groupby("split").mean(numeric_only=True).T.to_string())
    print(json.dumps(dup_stats, indent=2))

    # --- figure: share of test examples whose nearest training neighbour has sim >= x
    P.style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    xs = np.linspace(0.4, 1.0, 400)
    for name, col in (("Random", P.RASP), ("Cluster-aware", P.TEAL)):
        ys = np.array([[(m >= x).mean() * 100 for x in xs] for m in curves[name]])
        ax.plot(xs, ys.mean(0), color=col, lw=1.8, label=name)
        if len(ys) > 1:
            ax.fill_between(xs, ys.min(0), ys.max(0), color=col, alpha=0.2, lw=0)
    ax.axvline(0.8429, color=P.GREY, ls="--", lw=1)
    ax.text(0.8429, 3, r" $\tau$", color=P.INK, fontsize=8)
    ax.set_xlabel("Cosine similarity to the nearest training example")
    ax.set_ylabel("Test examples above, %")
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower left")
    P.save(fig, a.out_dir, "figS_leakage_nn")


if __name__ == "__main__":
    main()
