#!/usr/bin/env python3
"""
Error analysis (SUB / TAR) of the best configuration on the HDBSCAN cluster-aware
test split -> Table 2 / Table S7 and the error-taxonomy figure.

The script retrains one architecture with its best Optuna hyperparameters
(same split, same recipe as multi_seed_hdbscan.py), takes the word-level test
predictions and classifies every gold / predicted entity span:

  TP                   exact match (span and type)           -> strict TP
  boundary_truncated   same type, predicted span inside gold span
  boundary_extended    same type, gold span inside predicted span
  boundary_shifted     same type, partial overlap
  type_confusion       identical span, wrong type (SUB<->TAR)
  boundary_and_type    overlap, different span AND different type
  missed (FN)          gold span with no overlapping prediction
  spurious (FP)        predicted span with no overlapping gold span
  fragment             extra predicted piece overlapping an already matched gold span

Outputs in --save_dir:
  taxonomy_per_seed.csv      counts per seed x gold type
  taxonomy_summary.csv       mean +- s.d. over seeds (Table 2)
  per_type_prf.csv           strict P/R/F1 per entity type (from exact matches)
  recall_by_length.csv       strict recall by gold span length (words)
  errors_seed<S>.csv         every non-TP event with example text (for manual review)
  top_error_strings.csv      most frequent mis-handled strings
  error_taxonomy.png/.pdf    figure

Usage
-----
  python3 error_analysis_hdbscan.py \
      --data_path train.json \
      --vecs_cache embeddings.npy \
      --search_root experiments \
      --arch bert_bilstm_crf \
      --seeds 42 43 44 \
      --save_dir experiments/error_analysis
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

KINDS_GOLD = ["TP", "boundary_truncated", "boundary_extended", "boundary_shifted",
              "type_confusion", "boundary_and_type", "missed"]
KINDS_PRED = ["spurious", "fragment"]


def overlap(a: Tuple[int, int, str], b: Tuple[int, int, str]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def to_spans(tnc, tags: List[str]) -> List[Tuple[int, int, str]]:
    return [(d["start"], d["end"], d["label"]) for d in tnc.tags_to_spans(tags)]


def classify(gold: List[Tuple], pred: List[Tuple]) -> List[Dict]:
    """Return a list of events {kind, gold, pred}."""
    events: List[Dict] = []
    consumed = set()  # indices of preds already matched to a gold span
    pred_set = {p: i for i, p in enumerate(pred)}

    unmatched_gold = []
    for g in gold:
        if g in pred_set and pred_set[g] not in consumed:
            consumed.add(pred_set[g])
            events.append({"kind": "TP", "gold": g, "pred": g})
        else:
            unmatched_gold.append(g)

    for g in unmatched_gold:
        cands = [(overlap(g, p), i) for i, p in enumerate(pred) if i not in consumed and overlap(g, p) > 0]
        if not cands:
            events.append({"kind": "missed", "gold": g, "pred": None})
            continue
        _, i = max(cands)
        p = pred[i]
        consumed.add(i)
        same_label = p[2] == g[2]
        same_bounds = (p[0], p[1]) == (g[0], g[1])
        if same_bounds and not same_label:
            kind = "type_confusion"
        elif same_label:
            if p[0] >= g[0] and p[1] <= g[1]:
                kind = "boundary_truncated"
            elif p[0] <= g[0] and p[1] >= g[1]:
                kind = "boundary_extended"
            else:
                kind = "boundary_shifted"
        else:
            kind = "boundary_and_type"
        events.append({"kind": kind, "gold": g, "pred": p})

    for i, p in enumerate(pred):
        if i in consumed:
            continue
        kind = "fragment" if any(overlap(p, g) > 0 for g in gold) else "spurious"
        events.append({"kind": kind, "gold": None, "pred": p})
    return events


def align(tnc, test_ex, cfg, true_tags, pred_tags):
    """Match predicted sequences back to test examples (empty examples are skipped
    by decode_predictions, so align with a pointer)."""
    rows, k = [], 0
    for i, ex in enumerate(test_ex):
        tags = tnc.normalize_tags(ex["tags"], cfg.input_scheme, cfg.scheme)
        words, wtags = tnc._reconstruct_words_and_tags(ex["tokens"], tags)
        if k < len(true_tags):
            tt = true_tags[k]
            if len(tt) <= len(wtags) and wtags[:len(tt)] == tt:
                rows.append((i, words[:len(tt)], tt, pred_tags[k]))
                k += 1
    if k != len(true_tags):
        print(f"[warn] aligned {k} of {len(true_tags)} test sequences")
    return rows


def analyse(tnc, aligned, test_ex, types):
    gold_cnt = {t: Counter() for t in types}
    pred_cnt = {t: Counter() for t in types}
    len_tot, len_tp = Counter(), Counter()
    err_rows = []
    for i, words, tt, pp in aligned:
        gs, ps = to_spans(tnc, tt), to_spans(tnc, pp)
        for ev in classify(gs, ps):
            g, p, kind = ev["gold"], ev["pred"], ev["kind"]
            if g is not None:
                gold_cnt[g[2]][kind] += 1
                L = min(g[1] - g[0] + 1, 4)
                len_tot[(g[2], L)] += 1
                if kind == "TP":
                    len_tp[(g[2], L)] += 1
            else:
                pred_cnt[p[2]][kind] += 1
            if kind != "TP":
                err_rows.append({
                    "example_idx": i, "kind": kind,
                    "gold_type": g[2] if g else "", "gold_text": " ".join(words[g[0]:g[1] + 1]) if g else "",
                    "pred_type": p[2] if p else "", "pred_text": " ".join(words[p[0]:p[1] + 1]) if p else "",
                    "text": (test_ex[i].get("text") or " ".join(words))[:300],
                })
    return gold_cnt, pred_cnt, len_tot, len_tp, err_rows


def plot(summary: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = {"TP": "#54A24B", "boundary_truncated": "#F2CF5B", "boundary_extended": "#F58518",
           "boundary_shifted": "#E7BA52", "type_confusion": "#B279A2",
           "boundary_and_type": "#9D755D", "missed": "#E45756"}
    g = summary[summary["role"] == "gold"]
    types = list(g["type"].unique())
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.9 * len(types)))
    for r, t in enumerate(types):
        row = g[g["type"] == t].set_index("kind")["mean"]
        tot = row.sum()
        left = 0.0
        for k in KINDS_GOLD:
            v = row.get(k, 0.0) / tot * 100 if tot else 0
            ax.barh(r, v, left=left, color=pal[k], label=k if r == 0 else None)
            if v >= 3:
                ax.text(left + v / 2, r, f"{v:.0f}", ha="center", va="center", fontsize=8)
            left += v
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{t} (n={int(g[g['type'] == t]['mean'].sum())})" for t in types])
    ax.set_xlabel("% of gold entities")
    ax.set_xlim(0, 100)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Outcome of gold entities on the HDBSCAN test split")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"error_taxonomy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--save_dir", default="experiments/error_analysis")
    p.add_argument("--vecs_cache", type=Path, default=None)
    p.add_argument("--umap_cache", type=Path, default=None)
    p.add_argument("--search_root", default="experiments")
    p.add_argument("--params", default=None, help="explicit optuna_best.json (else auto-discovered)")
    p.add_argument("--arch", default="bert_bilstm_crf")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--donor_ckpt", default=None, help="needed only for --arch bert_lora")
    p.add_argument("--min_cluster_size", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=5)
    p.add_argument("--model_name", default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    import train_ner_cluster_aware as tnc
    from multi_seed_hdbscan import discover_params, backbone_from_ckpt

    save_dir = Path(os.path.expanduser(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = tnc.setup_logging(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pfile = args.params or discover_params(args.search_root, [args.arch], {})[args.arch][0]
    if not pfile:
        raise SystemExit(f"optuna_best.json for {args.arch} not found under {args.search_root}")
    best = json.load(open(pfile))
    logger.info(f"best params: {pfile}\n{best}")

    train_ex, dev_ex, test_ex = tnc.load_and_split(
        data_path=Path(os.path.expanduser(args.data_path)),
        train_ratio=0.8, val_ratio=0.1, logger=logger,
        embed_model="kamalkraj/BioSimCSE-BioLinkBERT-BASE", embed_device=None, embed_batch=64,
        vecs_cache=args.vecs_cache, umap_components=50, umap_neighbors=15, umap_min_dist=0.0,
        umap_cache=args.umap_cache, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples, cluster_selection_method="eom",
    )

    donor = None
    if args.arch == "bert_lora":
        if not args.donor_ckpt:
            raise SystemExit("bert_lora needs --donor_ckpt (bert_crf checkpoint)")
        donor = backbone_from_ckpt(args.donor_ckpt)

    base = dict(tnc.BASE_PARAMS)
    base.update({"model_name": args.model_name, "max_length": args.max_length,
                 "grad_accum_steps": args.grad_accum_steps, "epochs": args.epochs,
                 "num_workers": args.num_workers})
    ap = copy.deepcopy(base)
    ap.update(tnc.ARCHITECTURES[args.arch])
    ap.update(best)
    ap["architecture"] = args.arch
    cfg = tnc.TrainConfig(**ap)

    _orig_seed = tnc.seed_everything
    per_seed, prf_rows, len_rows = [], [], []
    top = Counter()

    for seed in args.seeds:
        logger.info(f"=== {args.arch} seed={seed} ===")
        tnc.seed_everything = lambda s=42, _s=seed: _orig_seed(_s)
        try:
            res = tnc.train_one_run(train_ex, dev_ex, test_ex, cfg, device, logger,
                                    backbone_state_dict=donor)
        finally:
            tnc.seed_everything = _orig_seed

        aligned = align(tnc, test_ex, cfg, res["test_seqeval"]["true_tags"], res["test_seqeval"]["pred_tags"])
        types = res["entity_types"]
        gold_cnt, pred_cnt, len_tot, len_tp, err_rows = analyse(tnc, aligned, test_ex, types)

        pd.DataFrame(err_rows).to_csv(save_dir / f"errors_seed{seed}.csv", index=False)
        for r in err_rows:
            top[(r["kind"], r["gold_type"] or r["pred_type"], (r["gold_text"] or r["pred_text"]).lower())] += 1

        for t in types:
            for k in KINDS_GOLD:
                per_seed.append({"seed": seed, "role": "gold", "type": t, "kind": k, "count": gold_cnt[t][k]})
            for k in KINDS_PRED:
                per_seed.append({"seed": seed, "role": "pred", "type": t, "kind": k, "count": pred_cnt[t][k]})
            tp = gold_cnt[t]["TP"]
            n_gold = sum(gold_cnt[t].values())
            n_pred = tp + sum(1 for r in err_rows if r["pred_type"] == t and r["kind"] != "TP")
            prec = tp / n_pred if n_pred else 0.0
            rec = tp / n_gold if n_gold else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            prf_rows.append({"seed": seed, "type": t, "tp": tp, "n_gold": n_gold, "n_pred": n_pred,
                             "precision": prec, "recall": rec, "f1": f1})
            for L in (1, 2, 3, 4):
                if len_tot[(t, L)]:
                    len_rows.append({"seed": seed, "type": t, "length_words": L if L < 4 else "4+",
                                     "n_gold": len_tot[(t, L)], "recall": len_tp[(t, L)] / len_tot[(t, L)]})
        logger.info(f"seed {seed}: seqeval strict F1 = {res['test_seqeval']['f1']:.4f} "
                    f"(span-level overall F1 from exact matches: "
                    f"{np.mean([r['f1'] for r in prf_rows if r['seed'] == seed]):.4f} macro)")

        res["model"].cpu()
        del res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(per_seed)
    df.to_csv(save_dir / "taxonomy_per_seed.csv", index=False)
    summ = (df.groupby(["role", "type", "kind"])["count"].agg(["mean", "std"]).reset_index())
    summ["std"] = summ["std"].fillna(0.0)
    summ.to_csv(save_dir / "taxonomy_summary.csv", index=False)

    prf = pd.DataFrame(prf_rows)
    prf.to_csv(save_dir / "per_type_prf_per_seed.csv", index=False)
    prf.groupby("type")[["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(save_dir / "per_type_prf.csv")
    pd.DataFrame(len_rows).groupby(["type", "length_words"])[["n_gold", "recall"]].mean().reset_index() \
        .to_csv(save_dir / "recall_by_length.csv", index=False)
    pd.DataFrame([{"kind": k, "type": t, "string": s, "count": c} for (k, t, s), c in top.most_common(100)]) \
        .to_csv(save_dir / "top_error_strings.csv", index=False)

    plot(summ, save_dir)
    print("\n=== Gold-entity outcomes (mean over seeds) ===")
    print(summ[summ["role"] == "gold"].pivot(index="type", columns="kind", values="mean").round(1).to_string())
    print("\n=== Strict P/R/F1 per type ===")
    print(prf.groupby("type")[["precision", "recall", "f1"]].mean().round(4).to_string())
    logger.info(f"Done: {save_dir}")


if __name__ == "__main__":
    main()
