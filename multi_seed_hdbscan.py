#!/usr/bin/env python3
"""
Retrain the BEST Optuna configuration of each architecture on the HDBSCAN
cluster-aware split with several training seeds, to obtain mean +- s.d.
for Table 1 / Table S5.

Why a separate script: train_ner_cluster_aware.py trains the final model with
a hard-coded seed (42) and therefore yields a single-seed result. This script
imports that module and re-runs only the final training step per seed. The
train/val/test split is built by the very same tnc.load_and_split() call
(UMAP random_state = 42, deterministic HDBSCAN), so it is identical for all
seeds -- only weight initialisation / data order / dropout vary.

Best hyperparameters are read from optuna_best.json files written by
train_ner_cluster_aware.py (experiments/<run>/<architecture>/optuna_best.json).

LoRA: warm-started from the backbone of a saved bert_crf checkpoint
(--donor_ckpt), the same donor for every LoRA seed, as in the main pipeline.

Usage
-----
  # 1) check which hyperparameter files will be used (no GPU needed)
  python3 multi_seed_hdbscan.py --search_root experiments --dry_run

  # 2) run
  python3 multi_seed_hdbscan.py \
      --data_path train.json \
      --vecs_cache embeddings.npy \
      --search_root experiments \
      --donor_ckpt experiments/run1/bert_crf/best_bert_crf.pt \
      --save_dir experiments/multi_seed
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_ARCHS = ["bert_linear", "bert_crf", "bert_bilstm_crf", "bert_lora"]


def discover_params(search_root: str, archs: List[str], overrides: Dict[str, str]):
    """For each architecture find the most recently modified optuna_best.json
    under search_root/**/<arch>/ (or use an explicit override)."""
    found = {}
    for arch in archs:
        if arch in overrides:
            found[arch] = (overrides[arch], [overrides[arch]])
            continue
        pattern = os.path.join(os.path.expanduser(search_root), "**", arch, "optuna_best.json")
        cands = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)
        found[arch] = (cands[0] if cands else None, cands)
    return found


def backbone_from_ckpt(path: str):
    import torch
    ck = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    backbone = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
    if not backbone:
        raise RuntimeError(f"No 'bert.*' keys in {path}")
    return backbone


def write_summary(rows: List[Dict], path: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns
                   if c.startswith("test_") or c in ("best_dev_f1", "best_epoch", "elapsed_min")]
    out = []
    for arch, g in df.groupby("architecture", sort=False):
        d = {"architecture": arch, "n_seeds": len(g),
             "n_trainable": int(g["n_trainable"].iloc[0]), "n_total": int(g["n_total"].iloc[0])}
        for c in metric_cols:
            d[c] = float(g[c].mean())
            d[c + "_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else float("nan")
        out.append(d)
    pd.DataFrame(out).to_csv(path, index=False)


def print_summary(path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(path)
    print("\n=== Summary (mean +- s.d. over seeds) ===")
    for _, r in df.iterrows():
        print(f"{r['architecture']:<18} n={int(r['n_seeds'])}  "
              f"strict F1 = {r.get('test_strict_f1', float('nan')):.4f} +- {r.get('test_strict_f1_std', float('nan')):.4f}   "
              f"seqeval F1 = {r.get('test_seqeval_f1', float('nan')):.4f} +- {r.get('test_seqeval_f1_std', float('nan')):.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=None)
    p.add_argument("--save_dir", default="experiments/multi_seed")
    p.add_argument("--vecs_cache", type=Path, default=None)
    p.add_argument("--umap_cache", type=Path, default=None)
    p.add_argument("--search_root", default="experiments")
    p.add_argument("--params", nargs="*", default=[],
                   help="explicit overrides: arch=path/to/optuna_best.json")
    p.add_argument("--architectures", nargs="+", default=DEFAULT_ARCHS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--donor_ckpt", default=None,
                   help="bert_crf checkpoint (best_bert_crf.pt) used to warm-start LoRA")
    p.add_argument("--lora_no_warmstart", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--min_cluster_size", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=5)
    p.add_argument("--model_name", default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = dict(kv.split("=", 1) for kv in args.params)
    found = discover_params(args.search_root, args.architectures, overrides)

    print("=== Hyperparameter files that will be used ===")
    missing = []
    for arch, (path, cands) in found.items():
        if path is None:
            print(f"{arch:<18} NOT FOUND under {args.search_root}")
            missing.append(arch)
            continue
        print(f"{arch:<18} {path}")
        if len(cands) > 1:
            print(f"{'':<18} (other candidates, older: {cands[1:]})")
        print(f"{'':<18} {json.load(open(path))}")
    if missing:
        raise SystemExit(f"Missing optuna_best.json for: {missing}. Use --params arch=path")
    if args.dry_run:
        return
    if not args.data_path:
        raise SystemExit("--data_path is required (unless --dry_run)")

    import pandas as pd
    import torch
    import train_ner_cluster_aware as tnc

    save_dir = Path(os.path.expanduser(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = tnc.setup_logging(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    (save_dir / "params_used.json").write_text(
        json.dumps({a: {"path": p, "params": json.load(open(p))} for a, (p, _) in found.items()},
                   indent=2), encoding="utf-8")

    train_ex, dev_ex, test_ex = tnc.load_and_split(
        data_path=Path(os.path.expanduser(args.data_path)),
        train_ratio=0.8, val_ratio=0.1, logger=logger,
        embed_model="kamalkraj/BioSimCSE-BioLinkBERT-BASE", embed_device=None, embed_batch=64,
        vecs_cache=args.vecs_cache, umap_components=50, umap_neighbors=15, umap_min_dist=0.0,
        umap_cache=args.umap_cache, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples, cluster_selection_method="eom",
    )

    donor = None
    if "bert_lora" in args.architectures:
        if args.donor_ckpt:
            donor = backbone_from_ckpt(args.donor_ckpt)
            logger.info(f"LoRA warm-start donor: {args.donor_ckpt}")
        elif not args.lora_no_warmstart:
            raise SystemExit("bert_lora requested: give --donor_ckpt (bert_crf checkpoint) "
                             "or pass --lora_no_warmstart explicitly.")

    base_params = dict(tnc.BASE_PARAMS)
    base_params.update({"model_name": args.model_name, "max_length": args.max_length,
                        "grad_accum_steps": args.grad_accum_steps, "epochs": args.epochs,
                        "num_workers": args.num_workers})

    _orig_seed = tnc.seed_everything

    def run_with_seed(seed, *a, **kw):
        # train_one_run() calls seed_everything(42) internally; redirect it.
        tnc.seed_everything = lambda s=42: _orig_seed(seed)
        try:
            return tnc.train_one_run(*a, **kw)
        finally:
            tnc.seed_everything = _orig_seed

    seed_csv = save_dir / "seed_runs.csv"
    rows: List[Dict] = pd.read_csv(seed_csv).to_dict("records") if seed_csv.exists() else []
    done = {(r["architecture"], int(r["seed"])) for r in rows}

    for arch in args.architectures:
        best = json.load(open(found[arch][0]))
        for seed in args.seeds:
            if (arch, seed) in done:
                logger.info(f"skip {arch} seed={seed} (already in seed_runs.csv)")
                continue
            logger.info("\n" + "=" * 80 + f"\n{arch}  seed={seed}\n" + "=" * 80)
            ap = copy.deepcopy(base_params)
            ap.update(tnc.ARCHITECTURES[arch])
            ap.update(best)
            ap["architecture"] = arch
            cfg = tnc.TrainConfig(**ap)

            t0 = time.time()
            result = run_with_seed(seed, train_ex, dev_ex, test_ex, cfg, device, logger,
                                   backbone_state_dict=donor if arch == "bert_lora" else None)
            model = result["model"]
            row = {
                "architecture": arch, "seed": seed,
                "n_total": sum(p.numel() for p in model.parameters()),
                "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "best_dev_f1": result["best_dev_f1"], "best_epoch": result["best_epoch"],
                "test_seqeval_f1": result["test_seqeval"].get("f1", float("nan")),
                "elapsed_min": (time.time() - t0) / 60.0,
            }
            for k, v in result["test_nervaluate"].items():
                row[f"test_{k}"] = v
            rows.append(row)
            logger.info(f"[{arch} seed={seed}] {row}")

            pd.DataFrame(rows).to_csv(seed_csv, index=False)
            write_summary(rows, save_dir / "summary.csv")

            model.cpu()
            del model, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_summary(save_dir / "summary.csv")
    logger.info(f"Done: {save_dir}")


if __name__ == "__main__":
    main()
