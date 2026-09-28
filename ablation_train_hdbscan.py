#!/usr/bin/env python3
"""
Ablation study on the HDBSCAN cluster-aware pipeline (train_ner_cluster_aware.py).

Mirrors the thesis's ablation_train.py structure and factor list, but reuses
the HDBSCAN-based clustering/training pipeline instead of MiniBatchKMeans, so
the ablations are directly comparable to the main HDBSCAN results (Table 1 /
Table S5) rather than to the old KMeans-based ablations (Table S8, thesis).

IMPORTANT: BEST_PARAMS below MUST be filled in with the REAL best
hyperparameters found by Optuna for bert_crf, bert_bilstm_crf and bert_lora
on the HDBSCAN split (see experiments/run1/<architecture>/optuna_best.json).
Only bert_linear's entry below is populated from an actual observed run; the
other three are placeholders and must not be trusted until replaced.

Ablation factors (same five groups as the thesis):
  1. Focal loss on/off       — bert_crf, bert_bilstm_crf
  2. BIO vs BIOES scheme     — bert_linear, bert_crf
  3. Equal vs adaptive class weights — bert_linear, bert_crf
  4. Layer pooling on/off    — bert_crf only
  5. LoRA warm-start vs random init — bert_lora (requires a donor bert_crf
     checkpoint; the donor is trained once here if not already present)

Usage
-----
  python3 ablation_train_hdbscan.py \
      --data_path train.json \
      --save_dir experiments/ablations_hdbscan \
      --vecs_cache embeddings.npy \
      --n_seeds 3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import train_ner_cluster_aware as tnc


# =============================================================================
# BEST_PARAMS — fill in from experiments/run1/<arch>/optuna_best.json
# =============================================================================

BEST_PARAMS: Dict[str, Dict] = {
    "bert_linear": {
        # REAL — observed Optuna best trial for bert_linear on the HDBSCAN split
        "batch_size": 6,
        "lr": 3.0292280459375563e-05,
        "weight_decay": 0.003925578644212777,
        "warmup_ratio": 0.06657837612652093,
        "dropout": 0.06531922701212732,
        "o_weight": 0.13047389542477017,
        "label_smoothing": 0.07539774284723448,
        "patience": 5,
    },
    "bert_crf": {
        # PLACEHOLDER — replace with real values from
        # experiments/run1/bert_crf/optuna_best.json before running ablations
        "batch_size": 4, "lr": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.1,
        "dropout": 0.15, "o_weight": 0.2, "label_smoothing": 0.05, "patience": 4,
        "use_layer_pooling": True, "focal_weight": 0.1, "focal_gamma": 2.0,
        "n_last_layers": 4, "mlp_hidden": 256,
    },
    "bert_bilstm_crf": {
        # PLACEHOLDER — replace with real values from
        # experiments/run1/bert_bilstm_crf/optuna_best.json before running ablations
        "batch_size": 4, "lr": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.1,
        "dropout": 0.15, "o_weight": 0.2, "label_smoothing": 0.05, "patience": 4,
        "lstm_hidden": 256, "lstm_layers": 1, "focal_weight": 0.1,
    },
    "bert_lora": {
        # PLACEHOLDER — replace with real values from
        # experiments/run1/bert_lora/optuna_best.json before running ablations
        "batch_size": 4, "lr": 3e-5, "weight_decay": 0.01, "warmup_ratio": 0.1,
        "dropout": 0.15, "o_weight": 0.2, "label_smoothing": 0.05, "patience": 4,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.1, "focal_weight": 0.1,
    },
}

PLACEHOLDER_ARCHS = {"bert_crf", "bert_bilstm_crf", "bert_lora"}


# =============================================================================
# Ablation definitions — same five groups as the thesis
# =============================================================================

ABLATIONS: List[Dict] = [
    # 1. Focal loss on/off
    {"name": "crf_focal_off", "arch": "bert_crf", "overrides": {"focal_weight": 0.0}},
    {"name": "crf_focal_on", "arch": "bert_crf", "overrides": {}},
    {"name": "bilstm_focal_off", "arch": "bert_bilstm_crf", "overrides": {"focal_weight": 0.0}},
    {"name": "bilstm_focal_on", "arch": "bert_bilstm_crf", "overrides": {}},

    # 2. Tagging scheme BIO vs BIOES
    {"name": "linear_bio", "arch": "bert_linear", "overrides": {}, "scheme": "BIO"},
    {"name": "linear_bioes", "arch": "bert_linear", "overrides": {}, "scheme": "BIOES"},
    {"name": "crf_bio", "arch": "bert_crf", "overrides": {}, "scheme": "BIO"},
    {"name": "crf_bioes", "arch": "bert_crf", "overrides": {}, "scheme": "BIOES"},

    # 3. Equal vs adaptive class weights
    {"name": "linear_equal_weights", "arch": "bert_linear", "overrides": {"equal_class_weights": True, "o_weight": 1.0}},
    {"name": "linear_adaptive_weights", "arch": "bert_linear", "overrides": {"equal_class_weights": False}},
    {"name": "crf_equal_weights", "arch": "bert_crf", "overrides": {"equal_class_weights": True, "o_weight": 1.0}},
    {"name": "crf_adaptive_weights", "arch": "bert_crf", "overrides": {"equal_class_weights": False}},

    # 4. Layer pooling on/off (bert_crf only)
    {"name": "crf_pooling_on", "arch": "bert_crf", "overrides": {"use_layer_pooling": True, "n_last_layers": 4}},
    {"name": "crf_pooling_off", "arch": "bert_crf", "overrides": {"use_layer_pooling": False, "n_last_layers": 1}},

    # 5. LoRA warm-start vs random init
    {"name": "lora_warmstart", "arch": "bert_lora", "overrides": {}, "lora_random_init": False},
    {"name": "lora_random", "arch": "bert_lora", "overrides": {}, "lora_random_init": True},
]


# =============================================================================
# equal_class_weights support (thesis's compute_class_weights extended)
# =============================================================================

def _patch_equal_class_weights() -> None:
    """Add an `equal_class_weights` flag to compute_class_weights, matching
    the thesis's ablation_train.py behaviour: when True, all classes
    (including O) get weight 1.0, disabling adaptive weighting entirely."""
    _orig = tnc.compute_class_weights

    def compute_class_weights_patched(examples, label2id, input_scheme, output_scheme,
                                       device, o_weight=0.1, equal_class_weights=False):
        if equal_class_weights:
            return torch.ones(len(label2id), dtype=torch.float32).to(device)
        return _orig(examples, label2id, input_scheme, output_scheme, device, o_weight=o_weight)

    tnc.compute_class_weights = compute_class_weights_patched


# =============================================================================
# Donor extraction for LoRA warm-start
# =============================================================================

def train_donor_and_extract_backbone(
    train_ex, dev_ex, test_ex, device, logger, save_dir: Path,
) -> Dict[str, torch.Tensor]:
    """Train bert_crf once with its best config and extract the BERT
    backbone, exactly as main() does in train_ner_cluster_aware.py, so that
    lora_warmstart uses a genuine task-tuned donor rather than the raw
    pretrained checkpoint."""
    logger.info("[donor] Training bert_crf once to extract warm-start backbone …")
    arch_params = copy.deepcopy(BASE_PARAMS_GLOBAL)
    arch_params.update(tnc.ARCHITECTURES["bert_crf"])
    arch_params.update(BEST_PARAMS["bert_crf"])
    arch_params["architecture"] = "bert_crf"
    cfg = tnc.TrainConfig(**arch_params)
    result = tnc.train_one_run(train_ex, dev_ex, test_ex, cfg, device, logger)
    backbone = tnc.extract_bert_backbone(result["model"])
    result["model"].cpu()
    del result["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return backbone


# =============================================================================
# Runner
# =============================================================================

BASE_PARAMS_GLOBAL: Dict = {}


def run_one_ablation(
    ablation: Dict,
    train_ex, dev_ex, test_ex,
    device, logger, save_dir: Path,
    n_seeds: int,
    donor_backbone: Optional[Dict[str, torch.Tensor]],
) -> Dict:
    arch = ablation["arch"]
    name = ablation["name"]

    arch_params = copy.deepcopy(BASE_PARAMS_GLOBAL)
    arch_params.update(tnc.ARCHITECTURES.get(arch, {}))
    arch_params.update(BEST_PARAMS[arch])
    arch_params["architecture"] = arch
    if "scheme" in ablation:
        arch_params["scheme"] = ablation["scheme"]
    arch_params.update(ablation["overrides"])

    equal_class_weights = arch_params.pop("equal_class_weights", False)
    lora_random_init = ablation.get("lora_random_init", None)

    backbone = None
    if arch == "bert_lora":
        backbone = None if lora_random_init else donor_backbone

    cfg = tnc.TrainConfig(**arch_params)

    seed_rows = []
    for i in range(n_seeds):
        seed = 42 + i
        tnc.seed_everything(seed)

        # monkey-patch compute_class_weights call site's kwarg for this run
        _orig_compute = tnc.compute_class_weights
        def compute_with_flag(*a, **kw):
            kw["equal_class_weights"] = equal_class_weights
            return _orig_compute(*a, **kw)
        tnc.compute_class_weights = compute_with_flag
        try:
            result = tnc.train_one_run(train_ex, dev_ex, test_ex, cfg, device, logger,
                                        backbone_state_dict=backbone)
        finally:
            tnc.compute_class_weights = _orig_compute

        row = {
            "ablation": name, "arch": arch, "seed": seed,
            "best_dev_f1": result["best_dev_f1"],
            "test_seqeval_f1": result["test_seqeval"].get("f1", float("nan")),
        }
        for k, v in result["test_nervaluate"].items():
            row[f"test_{k}"] = v
        seed_rows.append(row)

        result["model"].cpu()
        del result["model"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(seed_rows)
    df.to_csv(save_dir / f"{name}_seeds.csv", index=False)

    agg = {"ablation": name, "arch": arch, "n_seeds": n_seeds}
    for col in [c for c in df.columns if c not in ("ablation", "arch", "seed")]:
        agg[col] = float(df[col].mean())
        agg[f"{col}_std"] = float(df[col].std(ddof=1)) if n_seeds > 1 else 0.0
    return agg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--vecs_cache", type=Path, default=None)
    p.add_argument("--umap_cache", type=Path, default=None)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--ablations", nargs="+", default=None,
                   help="Subset of ablation names to run (default: all)")
    p.add_argument("--min_cluster_size", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=5)
    p.add_argument("--model_name", default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    global BASE_PARAMS_GLOBAL
    args = parse_args()
    save_dir = Path(os.path.expanduser(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = tnc.setup_logging(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if PLACEHOLDER_ARCHS:
        logger.warning(
            "!!! BEST_PARAMS for bert_crf / bert_bilstm_crf / bert_lora are "
            "PLACEHOLDERS, not real Optuna results. Replace them from "
            "experiments/run1/<arch>/optuna_best.json before trusting these "
            "ablation numbers. !!!"
        )

    _patch_equal_class_weights()

    BASE_PARAMS_GLOBAL = dict(tnc.BASE_PARAMS)
    BASE_PARAMS_GLOBAL.update({
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "gradient_checkpointing": True,
    })

    logger.info(f"Loading dataset: {args.data_path}")
    raw = tnc.load_json(Path(os.path.expanduser(args.data_path)))
    examples = tnc.clean_examples(raw)
    logger.info(f"Clean examples: {len(examples)}")

    clustered = tnc.cluster_from_embeddings(
        examples, embed_model="kamalkraj/BioSimCSE-BioLinkBERT-BASE",
        embed_device=None, embed_batch=64, vecs_cache=args.vecs_cache, logger=logger,
        umap_cache=args.umap_cache, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples, cluster_selection_method="eom",
    )
    train_ex, dev_ex, test_ex = tnc.cluster_split(clustered, 0.8, 0.1, logger=logger)
    train_ex = tnc._shuffle_list(train_ex)
    dev_ex = tnc._shuffle_list(dev_ex)
    test_ex = tnc._shuffle_list(test_ex)
    logger.info(f"Split: train={len(train_ex)} dev={len(dev_ex)} test={len(test_ex)}")

    requested = args.ablations or [a["name"] for a in ABLATIONS]
    to_run = [a for a in ABLATIONS if a["name"] in requested]

    needs_donor = any(a["name"] == "lora_warmstart" for a in to_run)
    donor_backbone = None
    if needs_donor:
        donor_backbone = train_donor_and_extract_backbone(train_ex, dev_ex, test_ex, device, logger, save_dir)

    all_results = []
    for ablation in to_run:
        logger.info("\n" + "=" * 80)
        logger.info(f"ABLATION: {ablation['name']}  (arch={ablation['arch']})")
        logger.info("=" * 80)
        agg = run_one_ablation(ablation, train_ex, dev_ex, test_ex, device, logger,
                                save_dir, args.n_seeds, donor_backbone)
        logger.info(f"[{ablation['name']}] {agg}")
        all_results.append(agg)

        pd.DataFrame(all_results).to_csv(save_dir / "ablation_summary.csv", index=False)

    (save_dir / "ablation_full.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(f"\nDone: {save_dir}")


if __name__ == "__main__":
    main()
