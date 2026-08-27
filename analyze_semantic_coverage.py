from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        "Missing dependency: sentence-transformers. Install with: pip install sentence-transformers"
    ) from e

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors

DEFAULT_MODEL = "kamalkraj/BioSimCSE-BioLinkBERT-BASE"

ASPECT_PATTERNS: Dict[str, Sequence[str]] = {
    "mechanism": [
        r"\bagonist\b", r"\bantagonist\b", r"\bactivat(?:e|ion|ed)\b",
        r"\binhibit(?:s|ed|ion|ing)?\b", r"\bbind(?:s|ing|er)?\b",
        r"\bstimulat(?:e|ion|ed|ing)\b", r"\bsuppress(?:es|ed|ion|ing)?\b",
        r"\bblock(?:s|ed|ing)?\b", r"\bmodulat(?:e|es|ed|ing|ion)\b",
        r"\brelease(?:s|d|ing)?\b", r"\bphosphorylat(?:e|es|ed|ion|ing)\b",
        r"\bproliferat(?:e|es|ed|ion|ing)\b", r"\bcytotox(?:ic|icity)\b",
        r"\bviability\b", r"\bexpression\b", r"\btranslocation\b",
    ],
    "readout": [
        r"\bic50\b", r"\bec50\b", r"\bki\b", r"\bkd\b",
        r"\bpercent(?:age)?\s+inhibition\b", r"\bfluorescence\b",
        r"\bluminescence\b", r"\bradioactivity\b", r"\belisa\b",
        r"\blc-?ms/?ms\b", r"\bassay\s+readout\b", r"\bresponse\b",
    ],
    "matrix": [
        r"\bcell\s*line\b", r"\bpbmc\b", r"\bhepatocyte\b",
        r"\bmicrosome\b", r"\bplasma\b", r"\btissue\b",
        r"\borganism\b", r"\bin\s+vitro\b", r"\bin\s+vivo\b",
        r"\bmembrane\b", r"\blysate\b",
    ],
    "target_cue": [
        r"\breceptor\b", r"\benzyme\b", r"\bkinase\b", r"\btransporter\b",
        r"\bchannel\b", r"\bprotein\b", r"\btarget\b", r"\bligand\b",
        r"\bcyp\d+[a-z]?\d*\b", r"\bcd\d+\b",
    ],
    "assay_type": [
        r"\bbinding\b", r"\bfunctional\b", r"\bphenotypic\b",
        r"\bcell-based\b", r"\bbiochemical\b", r"\benzymatic\b",
        r"\badme\b", r"\btoxicity\b", r"\bphysicochemical\b",
    ],
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("[cls]", " ").replace("[sep]", " ")
    text = text.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma")
    text = text.replace("–", "-").replace("—", "-").replace("/", " / ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_aspect_labels(text: str) -> List[str]:
    norm = normalize_text(text)
    labels: List[str] = []
    for aspect, patterns in ASPECT_PATTERNS.items():
        if any(re.search(pat, norm, flags=re.IGNORECASE) for pat in patterns):
            labels.append(aspect)
    return labels

def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)

@dataclass
class TextItem:
    text: str
    source_id: str
    meta: Dict[str, object]

def load_training_json(path: Path, text_field: str) -> List[TextItem]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    items: List[TextItem] = []
    for i, row in enumerate(data):
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        items.append(TextItem(text=text, source_id=f"train_{i}", meta=row.get("meta", {}) or {}))
    return items

def load_assays_from_sqlite(db_path: Path, table: str, text_column: str) -> List[TextItem]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            f"SELECT rowid AS rid, * FROM {table} WHERE {text_column} IS NOT NULL AND TRIM({text_column}) != ''"
        ).fetchall()
    finally:
        con.close()

    items: List[TextItem] = []
    for row in rows:
        text = str(row[text_column]).strip()
        meta = {k: row[k] for k in row.keys() if k not in {text_column}}
        source_id = str(row["assay_chembl_id"]) if "assay_chembl_id" in row.keys() else str(row["rid"])
        items.append(TextItem(text=text, source_id=source_id, meta=meta))
    return items

def build_encoder(model_name: str, device: Optional[str] = None) -> SentenceTransformer:
    kwargs = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)

def encode_texts(model: SentenceTransformer, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    embs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embs, dtype=np.float32)

def build_search_index(vectors: np.ndarray):
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if HAS_FAISS:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return ("faiss", index)
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(vectors)
    return ("sklearn", nn)

def knn_search(index_obj, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    backend, index = index_obj
    if backend == "faiss":
        sims, idx = index.search(vectors, k)
        return sims, idx
    distances, idx = index.kneighbors(vectors, n_neighbors=k, return_distance=True)
    sims = 1.0 - distances
    return sims.astype(np.float32), idx.astype(np.int64)

def top1_without_self(vectors: np.ndarray) -> np.ndarray:
    index_obj = build_search_index(vectors)
    sims, idx = knn_search(index_obj, vectors, k=min(2, len(vectors)))
    if len(vectors) == 1:
        return np.array([0.0], dtype=np.float32)
    out = np.zeros(len(vectors), dtype=np.float32)
    for i in range(len(vectors)):
        if idx[i, 0] == i and idx.shape[1] > 1:
            out[i] = sims[i, 1]
        else:
            out[i] = sims[i, 0]
    return out

def robust_quantile(x: np.ndarray, q: float, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return default
    return float(np.quantile(x, q))

def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    return {
        "n": float(scores.size),
        "mean": float(np.mean(scores)) if scores.size else 0.0,
        "median": float(np.median(scores)) if scores.size else 0.0,
        "p10": robust_quantile(scores, 0.10),
        "p25": robust_quantile(scores, 0.25),
        "p75": robust_quantile(scores, 0.75),
        "p90": robust_quantile(scores, 0.90),
        "min": float(np.min(scores)) if scores.size else 0.0,
        "max": float(np.max(scores)) if scores.size else 0.0,
    }

def format_summary(summary: Dict[str, float]) -> str:
    return (
        f"n={int(summary['n'])}, mean={summary['mean']:.3f}, median={summary['median']:.3f}, "
        f"p10={summary['p10']:.3f}, p25={summary['p25']:.3f}, p75={summary['p75']:.3f}, p90={summary['p90']:.3f}"
    )

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Assess semantic coverage of assay texts by a training corpus.")
    parser.add_argument("--train-json", type=Path, required=True, help="Training JSON with a 'text' field")
    parser.add_argument("--assays-db", type=Path, required=True, help="SQLite DB with assay descriptions")
    parser.add_argument("--assays-table", default="assays", help="SQLite table name (default: assays)")
    parser.add_argument("--assays-text-column", default="description", help="Assay text column (default: description)")
    parser.add_argument("--train-text-field", default="text", help="Training text field name (default: text)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Sentence embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest training neighbors to keep")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--sample-train-for-calibration", type=int, default=0,
                        help="Optional calibration subsample size for very large corpora (0 = use all)")
    parser.add_argument("--out-prefix", type=Path, default=Path("coverage_report"), help="Output prefix")
    args = parser.parse_args()

    train_items = load_training_json(args.train_json, args.train_text_field)
    assay_items = load_assays_from_sqlite(args.assays_db, args.assays_table, args.assays_text_column)

    if not train_items:
        raise SystemExit("No training texts found.")
    if not assay_items:
        raise SystemExit("No assay texts found.")

    print(f"Loaded {len(train_items):,} training texts and {len(assay_items):,} assay texts")
    print(f"Embedding model: {args.model}")

    model = build_encoder(args.model)

    train_texts = [normalize_text(x.text) for x in train_items]
    assay_texts = [normalize_text(x.text) for x in assay_items]

    train_vecs = encode_texts(model, train_texts, batch_size=args.batch_size)
    assay_vecs = encode_texts(model, assay_texts, batch_size=args.batch_size)

    if args.sample_train_for_calibration and len(train_vecs) > args.sample_train_for_calibration:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(train_vecs), size=args.sample_train_for_calibration, replace=False)
        calib_vecs = train_vecs[sample_idx]
    else:
        calib_vecs = train_vecs

    calib_nn = top1_without_self(calib_vecs)
    covered_thr = robust_quantile(calib_nn, 0.25, default=0.80)
    novel_thr = robust_quantile(calib_nn, 0.10, default=0.70)

    print("Calibration from training corpus:")
    print(f"  nearest-neighbor similarity summary: {format_summary(summarize_scores(calib_nn))}")
    print(f"  novel_thr   = {novel_thr:.3f}")
    print(f"  covered_thr = {covered_thr:.3f}")

    index_obj = build_search_index(train_vecs)
    k = max(1, args.top_k)
    sims, idx = knn_search(index_obj, assay_vecs, k=k)

    rows: List[Dict[str, object]] = []
    max_sims: List[float] = []
    final_scores: List[float] = []

    for i, assay in enumerate(tqdm(assay_items, desc="Scoring assays")):
        neighbor_sims = sims[i]
        neighbor_idx = idx[i]
        best_sim = float(neighbor_sims[0])
        best_train = train_items[int(neighbor_idx[0])]
        best_train_aspects = extract_aspect_labels(best_train.text)
        assay_aspects = extract_aspect_labels(assay.text)

        aspect_overlap = jaccard(assay_aspects, best_train_aspects)

        if covered_thr > novel_thr:
            sem_component = (best_sim - novel_thr) / (covered_thr - novel_thr)
        else:
            sem_component = best_sim
        sem_component = float(np.clip(sem_component, 0.0, 1.0))

        final_score = 0.85 * sem_component + 0.15 * aspect_overlap

        if best_sim >= covered_thr:
            bucket = "covered"
        elif best_sim >= novel_thr:
            bucket = "borderline"
        else:
            bucket = "novel"

        max_sims.append(best_sim)
        final_scores.append(final_score)

        row = {
            "assay_id": assay.source_id,
            "assay_text": assay.text,
            "best_train_id": best_train.source_id,
            "best_train_text": best_train.text,
            "best_cosine": round(best_sim, 6),
            "final_coverage_score": round(final_score, 6),
            "bucket": bucket,
            "assay_aspects": ",".join(assay_aspects),
            "best_train_aspects": ",".join(best_train_aspects),
            "aspect_overlap": round(aspect_overlap, 6),
            "topk_train_ids": ",".join(train_items[int(x)].source_id for x in neighbor_idx.tolist()),
            "topk_cosines": ",".join(f"{float(x):.4f}" for x in neighbor_sims.tolist()),
        }

        if assay.meta:
            row["assay_meta"] = json.dumps(assay.meta, ensure_ascii=False)
        rows.append(row)

    max_sims_arr = np.asarray(max_sims, dtype=np.float32)
    final_scores_arr = np.asarray(final_scores, dtype=np.float32)

    covered_rate = float(np.mean(max_sims_arr >= covered_thr))
    borderline_rate = float(np.mean((max_sims_arr >= novel_thr) & (max_sims_arr < covered_thr)))
    novel_rate = float(np.mean(max_sims_arr < novel_thr))

    print("\nCoverage summary")
    print(f"  covered   : {covered_rate:.1%}")
    print(f"  borderline: {borderline_rate:.1%}")
    print(f"  novel     : {novel_rate:.1%}")
    print(f"  raw best cosine summary: {format_summary(summarize_scores(max_sims_arr))}")
    print(f"  final score summary    : {format_summary(summarize_scores(final_scores_arr))}")

    out_prefix = args.out_prefix
    write_csv(out_prefix.with_suffix(".csv"), rows)
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": args.model,
                "training_count": len(train_items),
                "assay_count": len(assay_items),
                "calibration": {
                    "novel_thr": novel_thr,
                    "covered_thr": covered_thr,
                    "calib_nn_summary": summarize_scores(calib_nn),
                },
                "summary": {
                    "covered_rate": covered_rate,
                    "borderline_rate": borderline_rate,
                    "novel_rate": novel_rate,
                    "best_cosine_summary": summarize_scores(max_sims_arr),
                    "final_score_summary": summarize_scores(final_scores_arr),
                },
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    novel_order = np.argsort(max_sims_arr)
    print("\nMost novel assay texts:")
    for rank in novel_order[: min(10, len(novel_order))]:
        r = rows[int(rank)]
        print(f"  [{r['best_cosine']:.3f}] {r['assay_text'][:180]}")

    print(f"\nSaved: {out_prefix.with_suffix('.csv')}")
    print(f"Saved: {out_prefix.with_suffix('.json')}")

if __name__ == "__main__":
    main()
