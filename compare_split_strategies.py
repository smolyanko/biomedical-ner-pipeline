#!/usr/bin/env python3
"""
Compare data-splitting strategies to quantify data leakage from semantic
near-duplicates in the ChEMBL assay corpus.

REWRITE: previously this script clustered with MiniBatchKMeans (inherited
from the original split_comparison.py / train_ner_base.py lineage). It now
reuses the exact same UMAP + HDBSCAN clustering as train_ner_cluster_aware.py
(same function, same hyperparameters), so that the "cluster_stratified" and
"cluster_aware" strategies below operate on the SAME set of clusters and
differ ONLY in how those clusters are assigned to train/val/test. This
isolates the leakage effect from the choice of clustering algorithm, and
keeps the notion of "semantic cluster" consistent across the whole pipeline
(corpus expansion, final training split, and this leakage ablation).

Four strategies compared:
  01_random             — i.i.d. shuffle, ignores semantic structure entirely
  02_stratified_entity   — stratified by each example's dominant entity type
  03_cluster_stratified  — HDBSCAN clusters, but split WITHIN each cluster
                           (deliberately leakage-inducing: near-duplicates of
                           the same protocol template end up on both sides)
  04_cluster_aware       — HDBSCAN clusters, whole clusters assigned to one
                           split only (the honest, leakage-free strategy used
                           throughout the rest of the pipeline)

Architectures compared: bert_vanilla (unweighted linear head, no focal loss/
label smoothing/class weights — a plain baseline) plus the four architectures
defined in train_ner_cluster_aware.py (bert_linear, bert_crf,
bert_bilstm_crf, bert_lora).

Usage
-----
  python3 compare_split_strategies.py \
      --data_path train.json \
      --save_dir experiments/leakage \
      --vecs_cache embeddings.npy
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse everything from the cluster-aware training pipeline instead of
# duplicating models / tokenisation / metrics / Optuna wiring.
import train_ner_cluster_aware as tnc


# =============================================================================
# bert_vanilla — plain linear head, no class weighting / focal loss / label
# smoothing. Used only as a baseline reference in this leakage ablation.
# =============================================================================

class BertVanillaClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1, **_ignored):
        super().__init__()
        self.bert = tnc.AutoModel.from_pretrained(model_name, output_hidden_states=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, word_starts, word_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = tnc._gather_word_representations(out.last_hidden_state, word_starts, word_mask)
        logits = self.classifier(self.dropout(seq))
        preds = logits.argmax(-1).tolist()

        if labels is None:
            return {"predictions": preds, "mask": word_mask.bool()}

        valid = labels != tnc.LABEL_PAD_ID
        if not valid.any():
            return {"loss": logits.sum() * 0.0, "predictions": preds, "mask": word_mask.bool()}

        loss = F.cross_entropy(logits[valid], labels[valid])  # no weighting, no smoothing
        return {"loss": loss, "predictions": preds, "mask": word_mask.bool()}


def _patch_bert_vanilla() -> None:
    """Register bert_vanilla into the shared ARCHITECTURES / build_model dispatch
    from train_ner_cluster_aware.py, without modifying that file."""
    tnc.ARCHITECTURES["bert_vanilla"] = {
        "use_layer_pooling": False,
        "focal_weight": 0.0,
        "lstm_hidden": 0,
        "lstm_layers": 0,
    }

    _orig_build_model = tnc.build_model

    def build_model_with_vanilla(config, num_labels, class_weights, backbone_state_dict=None):
        if config.architecture == "bert_vanilla":
            return BertVanillaClassifier(config.model_name, num_labels, config.dropout)
        return _orig_build_model(config, num_labels, class_weights, backbone_state_dict=backbone_state_dict)

    tnc.build_model = build_model_with_vanilla


ALL_ARCHITECTURES = ["bert_vanilla"] + tnc.DEFAULT_ARCHITECTURES


# =============================================================================
# Splitting strategies
# =============================================================================

def _shuffle(items: List[Dict], seed: int = 42) -> List[Dict]:
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


def random_split(
    examples: List[Dict], train_ratio: float, val_ratio: float, seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Strategy 01: i.i.d. shuffle, ignores all semantic structure."""
    items = _shuffle(examples, seed=seed)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def _dominant_entity_type(ex: Dict) -> str:
    from collections import Counter
    counts = Counter()
    for tag in ex["tags"]:
        _, t = tnc.split_tag(tag)
        if t:
            counts[t] += 1
    if not counts:
        return "NONE"
    return counts.most_common(1)[0][0]


def stratified_entity_split(
    examples: List[Dict], train_ratio: float, val_ratio: float, seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Strategy 02: stratify by each example's dominant entity type, then
    split independently within each stratum. Still example-level random
    within a stratum, so semantic near-duplicates can still leak across
    splits — this strategy controls for label balance only, not leakage."""
    from collections import defaultdict
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        by_type[_dominant_entity_type(ex)].append(ex)

    train_all, val_all, test_all = [], [], []
    for typ, group in by_type.items():
        tr, va, te = random_split(group, train_ratio, val_ratio, seed=seed)
        train_all.extend(tr)
        val_all.extend(va)
        test_all.extend(te)
    return _shuffle(train_all, seed), _shuffle(val_all, seed), _shuffle(test_all, seed)


def cluster_stratified_split(
    clustered_examples: List[Dict], train_ratio: float, val_ratio: float, seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Strategy 03: same HDBSCAN clusters as cluster_aware_split, but split
    WITHIN each cluster proportionally — deliberately leakage-inducing, since
    near-duplicate examples from the same semantic cluster end up on both
    sides of the split. Included specifically to demonstrate the leakage
    effect against the honest strategy 04 below, using an IDENTICAL
    clustering (only the split-assignment rule differs)."""
    from collections import defaultdict
    by_cluster: Dict[int, List[Dict]] = defaultdict(list)
    for ex in clustered_examples:
        cid = ex.get("meta", {}).get("cluster", tnc.NOISE_CLUSTER_ID)
        by_cluster[cid].append(ex)

    train_all, val_all, test_all = [], [], []
    for cid, group in by_cluster.items():
        tr, va, te = random_split(group, train_ratio, val_ratio, seed=seed)
        train_all.extend(tr)
        val_all.extend(va)
        test_all.extend(te)
    return _shuffle(train_all, seed), _shuffle(val_all, seed), _shuffle(test_all, seed)


def cluster_aware_split(
    clustered_examples: List[Dict], train_ratio: float, val_ratio: float, seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Strategy 04: the honest strategy — identical to the split used in
    train_ner_cluster_aware.py. Whole HDBSCAN clusters (and all noise points)
    are assigned to exactly one split."""
    return tnc.cluster_split(clustered_examples, train_ratio, val_ratio, logger=None)


STRATEGIES = ["01_random", "02_stratified_entity", "03_cluster_stratified", "04_cluster_aware"]


def build_all_splits(
    examples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    embed_model: str,
    embed_device: Optional[str],
    embed_batch: int,
    vecs_cache: Optional[Path],
    umap_components: int,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_cache: Optional[Path],
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    seed: int,
    logger,
) -> Dict[str, Tuple[List[Dict], List[Dict], List[Dict]]]:
    # Cluster ONCE — reused identically by strategies 03 and 04.
    logger.info("Clustering corpus once (UMAP+HDBSCAN) for strategies 03/04 …")
    clustered = tnc.cluster_from_embeddings(
        examples,
        embed_model=embed_model,
        embed_device=embed_device,
        embed_batch=embed_batch,
        vecs_cache=vecs_cache,
        logger=logger,
        umap_components=umap_components,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        umap_cache=umap_cache,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )

    splits = {
        "01_random": random_split(examples, train_ratio, val_ratio, seed=seed),
        "02_stratified_entity": stratified_entity_split(examples, train_ratio, val_ratio, seed=seed),
        "03_cluster_stratified": cluster_stratified_split(clustered, train_ratio, val_ratio, seed=seed),
        "04_cluster_aware": cluster_aware_split(clustered, train_ratio, val_ratio, seed=seed),
    }

    for name, (tr, va, te) in splits.items():
        logger.info(f"[{name}] train={len(tr)} | val={len(va)} | test={len(te)}")

    return splits


# =============================================================================
# Runner
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compare data-splitting strategies (HDBSCAN-based leakage ablation)")

    p.add_argument("--data_path", required=True)
    p.add_argument("--save_dir", required=True)

    p.add_argument("--embed_model", default="kamalkraj/BioSimCSE-BioLinkBERT-BASE")
    p.add_argument("--embed_device", default=None)
    p.add_argument("--embed_batch", type=int, default=64)
    p.add_argument("--vecs_cache", type=Path, default=None)

    p.add_argument("--umap_components", type=int, default=50)
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.0)
    p.add_argument("--umap_cache", type=Path, default=None)

    p.add_argument("--min_cluster_size", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=5)
    p.add_argument("--cluster_selection_method", default="eom", choices=["eom", "leaf"])

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)

    p.add_argument("--architectures", nargs="+", default=ALL_ARCHITECTURES)
    p.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    p.add_argument("--optuna_trials", type=int, default=0,
                   help="0 = train once with BASE_PARAMS defaults per (strategy, architecture); "
                        ">0 = run Optuna search per (strategy, architecture) — expensive, O(strategies*architectures)")
    p.add_argument("--selection_metric", default="strict_f1")
    p.add_argument("--fresh_optuna", action="store_true", default=False)

    for k, v in tnc.BASE_PARAMS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(f"--{k}", type=float, default=v)
        else:
            p.add_argument(f"--{k}", default=v)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(os.path.expanduser(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = tnc.setup_logging(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    _patch_bert_vanilla()

    logger.info(f"Loading dataset: {args.data_path}")
    raw = tnc.load_json(Path(os.path.expanduser(args.data_path)))
    examples = tnc.clean_examples(raw)
    logger.info(f"Clean examples: {len(examples)}")

    splits_by_strategy = build_all_splits(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
        embed_batch=args.embed_batch,
        vecs_cache=args.vecs_cache,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_cache=args.umap_cache,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        seed=42,
        logger=logger,
    )

    base_params = {k: getattr(args, k) for k in tnc.BASE_PARAMS}

    all_rows: List[Dict] = []
    csv_path = save_dir / "comparison_partial.csv"

    for strategy in args.strategies:
        train_ex, dev_ex, test_ex = splits_by_strategy[strategy]

        for architecture in args.architectures:
            logger.info("\n" + "=" * 80)
            logger.info(f"STRATEGY={strategy}  ARCHITECTURE={architecture}")
            logger.info("=" * 80)

            arch_params = copy.deepcopy(base_params)
            arch_params.update(tnc.ARCHITECTURES.get(architecture, {}))
            arch_params["architecture"] = architecture

            if args.optuna_trials > 0:
                study = tnc.run_optuna(
                    architecture, train_ex, dev_ex,
                    tnc.TrainConfig(**arch_params),
                    device, logger, save_dir / strategy,
                    args.optuna_trials, fresh=args.fresh_optuna,
                )
                arch_params.update(study.best_params)

            final_cfg = tnc.TrainConfig(**arch_params)
            result = tnc.train_one_run(train_ex, dev_ex, test_ex, final_cfg, device, logger)

            row = {
                "strategy": strategy,
                "architecture": architecture,
                "n_train": len(train_ex),
                "n_val": len(dev_ex),
                "n_test": len(test_ex),
                "best_dev_f1": result["best_dev_f1"],
                "test_seqeval_f1": result["test_seqeval"].get("f1", float("nan")),
            }
            for k, v in result["test_nervaluate"].items():
                row[f"test_{k}"] = v

            logger.info(f"[{strategy}/{architecture}] {row}")
            all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(csv_path, index=False)

            result["model"].cpu()
            del result["model"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(save_dir / "comparison_full.csv", index=False)

    metric_col_map = {
        "strict_f1": "test_strict_f1", "partial_f1": "test_partial_f1",
        "exact_f1": "test_exact_f1", "ent_type_f1": "test_ent_type_f1",
        "f1": "test_seqeval_f1",
    }
    sel_col = metric_col_map.get(args.selection_metric, f"test_{args.selection_metric}")

    if sel_col in df.columns:
        pivot = df.pivot_table(index="strategy", columns="architecture", values=sel_col)
        pivot = pivot.reindex(STRATEGIES)
        pivot.to_csv(save_dir / f"pivot_{sel_col}.csv")
        logger.info(f"\nPivot table ({sel_col}):\n{pivot.to_string()}")

        best_per_strategy = {}
        for strategy in STRATEGIES:
            if strategy not in pivot.index:
                continue
            row = pivot.loc[strategy].dropna()
            if row.empty:
                continue
            best_arch = row.idxmax()
            best_per_strategy[strategy] = {"architecture": best_arch, sel_col: float(row[best_arch])}
        (save_dir / "best_by_strategy.json").write_text(
            json.dumps(best_per_strategy, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        logger.info(f"\nBest architecture per strategy ({sel_col}):")
        for s, info in best_per_strategy.items():
            logger.info(f"  {s}: {info['architecture']} ({sel_col}={info[sel_col]:.4f})")

    logger.info(f"\nDone: {save_dir}")


if __name__ == "__main__":
    main()
