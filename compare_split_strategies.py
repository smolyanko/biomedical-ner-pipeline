import argparse
import copy
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import train_ner_base
from train_ner_base import (
    DEFAULT_ARCHITECTURES,
    LABEL_PAD_ID,
    TrainConfig,
    clean_examples,
    cluster_from_embeddings,
    cluster_split as cluster_aware_split_original,
    compute_class_weights,
    extract_entity_types,
    final_evaluate,
    load_json,
    prepare_items_retokenize,
    seed_everything,
    setup_logging,
    split_tag,
    train_one_run,
    _shuffle_list,
    _validate_ratios,
    _gather_word_representations,
)

class BertVanillaClassifier(nn.Module):

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, word_starts, word_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = _gather_word_representations(out.last_hidden_state, word_starts, word_mask)
        logits = self.classifier(self.dropout(seq))
        preds = logits.argmax(-1).tolist()

        if labels is None:
            return {"predictions": preds, "mask": word_mask.bool()}

        valid = labels != LABEL_PAD_ID
        if not valid.any():
            return {"loss": logits.sum() * 0.0, "predictions": preds, "mask": word_mask.bool()}

        loss = F.cross_entropy(logits[valid], labels[valid])
        return {"loss": loss, "predictions": preds, "mask": word_mask.bool()}

LOCAL_ARCHITECTURES = copy.deepcopy(train_ner_base.ARCHITECTURES)
LOCAL_ARCHITECTURES["bert_vanilla"] = {
    "use_layer_pooling": False,
    "focal_weight": 0.0,
    "lstm_hidden": 0,
    "lstm_layers": 0,
}

train_ner_base.ARCHITECTURES = LOCAL_ARCHITECTURES

_original_build_model = train_ner_base.build_model

def _build_model_extended(
    config: TrainConfig,
    num_labels: int,
    class_weights: torch.Tensor,
    backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    if config.architecture == "bert_vanilla":
        return BertVanillaClassifier(config.model_name, num_labels, config.dropout)
    return _original_build_model(config, num_labels, class_weights, backbone_state_dict)

train_ner_base.build_model = _build_model_extended

def random_split(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    _validate_ratios(train_ratio, val_ratio)
    rng = random.Random(42)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    if logger:
        logger.info(f"[random] split: train={len(train)} | val={len(val)} | test={len(test)}")
    return train, val, test

def stratified_split_by_dominant_entity(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    _validate_ratios(train_ratio, val_ratio)
    rng = random.Random(42)
    groups = defaultdict(list)
    for ex in examples:
        tags = ex["tags"]
        types = [split_tag(t)[1] for t in tags if split_tag(t)[1]]
        dominant = Counter(types).most_common(1)[0][0] if types else "NONE"
        groups[dominant].append(ex)

    train, val, test = [], [], []
    for group_exs in groups.values():
        rng.shuffle(group_exs)
        n = len(group_exs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = 1
        train.extend(group_exs[:n_train])
        val.extend(group_exs[n_train : n_train + n_val])
        test.extend(group_exs[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    if logger:
        logger.info(f"[stratified_entity] split: train={len(train)} | val={len(val)} | test={len(test)}")
    return train, val, test

def cluster_stratified_split(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    _validate_ratios(train_ratio, val_ratio)
    rng = random.Random(42)
    by_cluster = defaultdict(list)
    for ex in examples:
        cid = ex.get("meta", {}).get("cluster", -1)
        by_cluster[cid].append(ex)

    train, val, test = [], [], []
    for exs in by_cluster.values():
        rng.shuffle(exs)
        n = len(exs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = 1
        train.extend(exs[:n_train])
        val.extend(exs[n_train : n_train + n_val])
        test.extend(exs[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    if logger:
        logger.info(
            f"[cluster_stratified] LEAKAGE split: train={len(train)} | val={len(val)} | test={len(test)} "
            f"(split WITHIN {len(by_cluster)} clusters)"
        )
    return train, val, test

def cluster_aware_split_wrapper(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    return cluster_aware_split_original(examples, train_ratio, val_ratio, logger=logger)

def run_single_experiment(
    strategy_name: str,
    train_ex: List[Dict],
    dev_ex: List[Dict],
    test_ex: List[Dict],
    architecture: str,
    base_config: TrainConfig,
    device: torch.device,
    logger: logging.Logger,
    save_dir: Path,
) -> Dict:
    cfg_dict = copy.deepcopy(base_config.__dict__)
    cfg_dict.update(LOCAL_ARCHITECTURES[architecture])
    cfg_dict["architecture"] = architecture
    cfg = TrainConfig(**cfg_dict)

    logger.info(f"\n{'='*70}")
    logger.info(f"Strategy : {strategy_name}")
    logger.info(f"Architecture : {architecture}")
    logger.info(f"{'='*70}")

    result = train_one_run(
        train_examples=train_ex,
        dev_examples=dev_ex,
        test_examples=test_ex,
        config=cfg,
        device=device,
        logger=logger,
        trial=None,
        backbone_state_dict=None,
    )

    row = {
        "strategy": strategy_name,
        "architecture": architecture,
        "best_dev_f1": result["best_dev_f1"],
        "best_epoch": result["best_epoch"],
        "test_seqeval_f1": result["test_seqeval"].get("f1", float("nan")),
        "test_seqeval_precision": result["test_seqeval"].get("precision", float("nan")),
        "test_seqeval_recall": result["test_seqeval"].get("recall", float("nan")),
    }
    for k, v in result["test_nervaluate"].items():
        row[f"test_{k}"] = v

    arch_dir = save_dir / strategy_name / architecture
    arch_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": result["model"].state_dict(),
            "label2id": result["label2id"],
            "id2label": result["id2label"],
            "entity_types": result["entity_types"],
            "config": cfg.__dict__,
            "metrics": row,
        },
        arch_dir / f"best_{architecture}.pt",
    )
    (arch_dir / "metrics.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    del result["model"]
    torch.cuda.empty_cache()
    return row

def main():
    p = argparse.ArgumentParser(
        description="Comparison of splitting strategies with data leakage simulation"
    )
    p.add_argument("--data_path", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--n_clusters", type=int, default=100)
    p.add_argument("--embed_model", default="kamalkraj/BioSimCSE-BioLinkBERT-BASE")
    p.add_argument("--embed_device", default=None)
    p.add_argument("--embed_batch", type=int, default=64)
    p.add_argument("--vecs_cache", type=Path, default=None)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument(
        "--architectures",
        nargs="+",
        default=["bert_vanilla"] + DEFAULT_ARCHITECTURES,
        choices=sorted(LOCAL_ARCHITECTURES.keys()),
    )
    p.add_argument("--selection_metric", default="strict_f1")
    p.add_argument("--model_name", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    save_dir = Path(os.path.expanduser(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    seed_everything(args.seed)

    logger.info("Loading & clustering data …")
    raw = load_json(Path(os.path.expanduser(args.data_path)))
    clean = clean_examples(raw)
    logger.info(f"Clean examples: {len(clean)}")

    clustered = cluster_from_embeddings(
        clean,
        n_clusters=args.n_clusters,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
        embed_batch=args.embed_batch,
        vecs_cache=args.vecs_cache,
        logger=logger,
    )

    strategies = {
        "01_random": lambda exs: random_split(exs, args.train_ratio, args.val_ratio, logger),
        "02_stratified_entity": lambda exs: stratified_split_by_dominant_entity(
            exs, args.train_ratio, args.val_ratio, logger
        ),
        "03_cluster_stratified": lambda exs: cluster_stratified_split(
            exs, args.train_ratio, args.val_ratio, logger
        ),
        "04_cluster_aware": lambda exs: cluster_aware_split_wrapper(
            exs, args.train_ratio, args.val_ratio, logger
        ),
    }

    base_cfg = TrainConfig(
        architecture="bert_linear",
        model_name=args.model_name,
        input_scheme="BIO",
        scheme="BIOES",
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        dropout=args.dropout,
        patience=args.patience,
        use_amp=args.use_amp,
    )

    all_results: List[Dict] = []

    for strategy_name, split_fn in strategies.items():
        logger.info(f"\n{'#'*80}")
        logger.info(f"# STRATEGY: {strategy_name}")
        logger.info(f"{'#'*80}")

        train_ex, dev_ex, test_ex = split_fn(clustered)
        train_ex = _shuffle_list(train_ex)
        dev_ex = _shuffle_list(dev_ex)
        test_ex = _shuffle_list(test_ex)

        for arch in args.architectures:
            try:
                row = run_single_experiment(
                    strategy_name=strategy_name,
                    train_ex=train_ex,
                    dev_ex=dev_ex,
                    test_ex=test_ex,
                    architecture=arch,
                    base_config=base_cfg,
                    device=device,
                    logger=logger,
                    save_dir=save_dir,
                )
                all_results.append(row)
                pd.DataFrame(all_results).to_csv(
                    save_dir / "comparison_partial.csv", index=False
                )
            except Exception as e:
                logger.error(f"FAILED {strategy_name}/{arch}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                all_results.append(
                    {
                        "strategy": strategy_name,
                        "architecture": arch,
                        "best_dev_f1": float("nan"),
                        "test_seqeval_f1": float("nan"),
                        "error": str(e),
                    }
                )

    df = pd.DataFrame(all_results)
    df.to_csv(save_dir / "comparison_final.csv", index=False)

    pivot = df.pivot_table(
        index="strategy",
        columns="architecture",
        values="test_seqeval_f1",
        aggfunc="first",
    )
    pivot.to_csv(save_dir / "comparison_pivot_seqeval_f1.csv")

    logger.info(f"\n{'='*80}")
    logger.info("FINAL COMPARISON TABLE (seqeval F1)")
    logger.info(f"{'='*80}")
    logger.info("\n" + pivot.to_string())

    metric_col = f"test_{args.selection_metric}"
    if metric_col in df.columns:
        best_per_strategy = df.loc[df.groupby("strategy")[metric_col].idxmax()]
        logger.info(f"\nBest by strategy ({metric_col}):")
        for _, row in best_per_strategy.iterrows():
            logger.info(
                f"  {row['strategy']}: {row['architecture']} = {row[metric_col]:.4f}"
            )

    logger.info(f"\nAll results saved to: {save_dir}")

if __name__ == "__main__":
    main()
