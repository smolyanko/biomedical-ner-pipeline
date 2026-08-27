from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit("pip install sentence-transformers") from e

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.decomposition import PCA

try:
    import hdbscan as hdbscan_lib
    HAS_HDBSCAN_PKG = True
except ImportError:
    HAS_HDBSCAN_PKG = False
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
    except ImportError:
        raise SystemExit(
            "HDBSCAN not found.  You have two options:\n"
            "  1. pip install hdbscan  (needs python3-devel + gcc)\n"
            "  2. pip install -U scikit-learn  (needs >= 1.3, you have 1.6 — should work)"
        )

try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise SystemExit("pip install transformers") from e

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors

@dataclass
class TextItem:
    text: str
    source_id: str
    meta: Dict

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("[cls]", " ").replace("[sep]", " ")
    text = text.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma")
    text = text.replace("–", "-").replace("—", "-").replace("/", " / ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_training_json(path: Path, text_field: str) -> List[TextItem]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    items: List[TextItem] = []
    for i, row in enumerate(data):
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        items.append(TextItem(
            text=text,
            source_id=f"train_{i}",
            meta=row.get("meta", {}) or {}
        ))
    return items

def load_assays_from_sqlite(db_path: Path, table: str, text_col: str) -> List[TextItem]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            f"SELECT rowid AS rid, * FROM {table} "
            f"WHERE {text_col} IS NOT NULL AND TRIM({text_col}) != ''"
        ).fetchall()
    finally:
        con.close()
    items: List[TextItem] = []
    for row in rows:
        text = str(row[text_col]).strip()
        meta = {k: row[k] for k in row.keys() if k != text_col}
        source_id = (
            str(row["assay_chembl_id"])
            if "assay_chembl_id" in row.keys()
            else str(row["rid"])
        )
        items.append(TextItem(text=text, source_id=source_id, meta=meta))
    return items

def build_encoder(model_name: str, device: Optional[str] = None) -> SentenceTransformer:
    kwargs: Dict = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)

def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 64,
    desc: str = "Encoding",
) -> np.ndarray:
    embs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embs, dtype=np.float32)

def initial_coverage_scores(
    train_vecs: np.ndarray,
    assay_vecs: np.ndarray,
    batch: int = 2048,
) -> np.ndarray:
    N = len(assay_vecs)
    d_max = np.full(N, -1.0, dtype=np.float32)

    if HAS_FAISS:
        print("Building FAISS index for initial coverage …")
        index = faiss.IndexFlatIP(train_vecs.shape[1])
        index.add(train_vecs)
        sims, _ = index.search(assay_vecs, k=1)
        d_max = sims[:, 0].astype(np.float32)
    else:
        print("Using numpy batched matmul for initial coverage …")
        for start in tqdm(range(0, N, batch), desc="Initial coverage"):
            chunk = assay_vecs[start : start + batch]
            sims = chunk @ train_vecs.T
            d_max[start : start + batch] = sims.max(axis=1)

    return d_max

def reduce_umap(
    vecs: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    random_state: int = 42,
) -> np.ndarray:
    if HAS_UMAP:
        print(f"UMAP: {vecs.shape} → ({len(vecs)}, {n_components}) …")
        t0 = time.time()
        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
            low_memory=True,
            verbose=True,
        )
        reduced = reducer.fit_transform(vecs)
        print(f"UMAP done in {time.time() - t0:.0f}s")
    else:
        n_components = min(n_components, vecs.shape[1], len(vecs) - 1)
        print(f"umap-learn not installed — using PCA fallback: "
              f"{vecs.shape} → ({len(vecs)}, {n_components}) …")
        print("  (install umap-learn for better cluster quality)")
        t0 = time.time()
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(vecs)
        print(f"PCA done in {time.time() - t0:.0f}s")
    return reduced.astype(np.float32)

def run_hdbscan(
    reduced: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",
) -> np.ndarray:
    print(f"HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples} …")
    if HAS_HDBSCAN_PKG:
        print("  backend: hdbscan package")
    else:
        print("  backend: sklearn.cluster.HDBSCAN (sklearn 1.3+)")
    t0 = time.time()

    if HAS_HDBSCAN_PKG:
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",
            core_dist_n_jobs=-1,
            prediction_data=True,
        )
    else:
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",
            n_jobs=-1,
        )

    labels = clusterer.fit_predict(reduced)
    n_clusters = int(labels.max()) + 1
    n_noise    = int(np.sum(labels == -1))
    print(
        f"HDBSCAN done in {time.time() - t0:.0f}s  "
        f"→ {n_clusters} clusters, {n_noise} noise points "
        f"({n_noise / len(labels):.1%})"
    )
    return labels.astype(np.int32)

def find_medoids(
    vecs: np.ndarray,
    labels: np.ndarray,
    n_medoids: int = 1,
) -> Dict[int, List[int]]:
    cluster_ids = sorted(set(labels[labels >= 0]))
    medoids: Dict[int, List[int]] = {}

    for cid in tqdm(cluster_ids, desc="Finding medoids"):
        mask = labels == cid
        idx  = np.where(mask)[0]
        cvecs = vecs[idx]

        centroid = cvecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid /= norm

        sims = cvecs @ centroid
        top  = np.argsort(sims)[::-1][:n_medoids]
        medoids[cid] = [int(idx[i]) for i in top]

    return medoids

def tokenize_for_ner(
    text: str,
    tokenizer,
    max_length: int = 512,
) -> Dict:
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=False,
        add_special_tokens=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    tags   = ["O"] * len(tokens)
    return {
        "tokens":          tokens,
        "tags":            tags,
        "input_ids":       enc["input_ids"],
        "attention_mask":  enc["attention_mask"],
    }

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Cluster uncovered assays (UMAP+HDBSCAN) and extract medoids "
            "for NER annotation.  Output is compatible with "
            "biomedbert_ready_data_dedup.json."
        )
    )

    grp_data = ap.add_argument_group("Raw data inputs")
    grp_data.add_argument("--train-json",          type=Path)
    grp_data.add_argument("--train-text-field",    default="text")
    grp_data.add_argument("--assays-db",           type=Path)
    grp_data.add_argument("--assays-table",        default="assays")
    grp_data.add_argument("--assays-text-column",  default="description")
    grp_data.add_argument("--model",               default="kamalkraj/BioSimCSE-BioLinkBERT-BASE",
                          help="Sentence embedding model")
    grp_data.add_argument("--tokenizer",           default=None,
                          help="Tokenizer path/name for NER format output "
                               "(default: same as --model)")
    grp_data.add_argument("--batch-size",          type=int, default=64)
    grp_data.add_argument("--device",              default=None)

    grp_cache = ap.add_argument_group("Embedding cache")
    grp_cache.add_argument("--train-vecs-cache",   type=Path)
    grp_cache.add_argument("--assay-vecs-cache",   type=Path)
    grp_cache.add_argument("--assay-ids-cache",    type=Path)
    grp_cache.add_argument("--save-train-vecs",    type=Path)
    grp_cache.add_argument("--save-assay-vecs",    type=Path)
    grp_cache.add_argument("--save-assay-ids",     type=Path)

    grp_cov = ap.add_argument_group("Coverage thresholds")
    grp_cov.add_argument("--covered-thr",          type=float, default=0.8429,
                         help="Assays with sim < this are 'uncovered' (default from report)")

    grp_umap = ap.add_argument_group("UMAP parameters")
    grp_umap.add_argument("--umap-components",     type=int, default=50)
    grp_umap.add_argument("--umap-neighbors",      type=int, default=15)
    grp_umap.add_argument("--umap-cache",          type=Path,
                          help="Cache for UMAP-reduced vectors (.npy)")

    grp_hdb = ap.add_argument_group("HDBSCAN parameters")
    grp_hdb.add_argument("--min-cluster-size",     type=int, default=30,
                         help="Min assays per cluster (smaller = more fine-grained types)")
    grp_hdb.add_argument("--min-samples",          type=int, default=5)
    grp_hdb.add_argument("--cluster-method",       default="eom",
                         choices=["eom", "leaf"],
                         help="eom=fewer larger clusters, leaf=more granular")
    grp_hdb.add_argument("--n-examples",           type=int, default=5,
                         help="Central examples per cluster (default 5). "
                              "BERT generalizes better with 5-10 varied examples per type.")
    grp_hdb.add_argument("--top-clusters",         type=int, default=0,
                         help="Only take top N clusters by size (0=all). "
                              "Recommended: 200 for first annotation round.")
    grp_hdb.add_argument("--save-labels",          type=Path,
                         help="Save HDBSCAN labels (.npy) to skip recomputation next run.")
    grp_hdb.add_argument("--load-labels",          type=Path,
                         help="Load saved HDBSCAN labels (.npy) — skips UMAP+HDBSCAN entirely.")

    grp_out = ap.add_argument_group("Output")
    grp_out.add_argument("--out-prefix",           type=Path,
                         default=Path("logs/annotation_candidates"))
    grp_out.add_argument("--max-length",           type=int, default=512,
                         help="Max token length (must match your BERT config)")

    args = ap.parse_args()

    model = None

    if args.train_vecs_cache and args.train_vecs_cache.exists():
        print(f"Loading cached train vecs: {args.train_vecs_cache}")
        train_vecs = np.load(str(args.train_vecs_cache)).astype(np.float32)
        norms = np.linalg.norm(train_vecs, axis=1, keepdims=True)
        train_vecs /= np.maximum(norms, 1e-9)
    elif args.train_json:
        print(f"Loading training data: {args.train_json}")
        train_items_raw = load_training_json(args.train_json, args.train_text_field)
        if not train_items_raw:
            raise SystemExit("No training texts found.")
        model = build_encoder(args.model, args.device)
        train_texts = [normalize_text(x.text) for x in train_items_raw]
        train_vecs = encode_texts(model, train_texts, batch_size=args.batch_size,
                                  desc="Encoding training")
        if args.save_train_vecs:
            np.save(str(args.save_train_vecs), train_vecs)
            print(f"Saved train vecs → {args.save_train_vecs}")
    else:
        raise SystemExit("Provide --train-json or --train-vecs-cache.")

    if not args.assays_db:
        raise SystemExit("--assays-db is required to access assay texts.")
    print(f"Loading assay texts from {args.assays_db} …")
    assay_items = load_assays_from_sqlite(
        args.assays_db, args.assays_table, args.assays_text_column
    )
    if not assay_items:
        raise SystemExit("No assay texts found.")
    print(f"  {len(assay_items):,} assays loaded")

    if args.assay_vecs_cache and args.assay_vecs_cache.exists():
        print(f"Loading cached assay vecs: {args.assay_vecs_cache}")
        assay_vecs = np.load(str(args.assay_vecs_cache)).astype(np.float32)
        norms = np.linalg.norm(assay_vecs, axis=1, keepdims=True)
        assay_vecs /= np.maximum(norms, 1e-9)
        assert len(assay_vecs) == len(assay_items), (
            f"Cache size mismatch: {len(assay_vecs)} vecs vs {len(assay_items)} texts. "
            "Re-run without cache to recompute."
        )
    else:
        if model is None:
            model = build_encoder(args.model, args.device)
        assay_texts = [normalize_text(x.text) for x in assay_items]
        assay_vecs = encode_texts(model, assay_texts, batch_size=args.batch_size,
                                  desc="Encoding assays")
        if args.save_assay_vecs:
            np.save(str(args.save_assay_vecs), assay_vecs)
            print(f"Saved assay vecs → {args.save_assay_vecs}")
        if args.save_assay_ids:
            args.save_assay_ids.parent.mkdir(parents=True, exist_ok=True)
            with args.save_assay_ids.open("w", encoding="utf-8") as fh:
                for item in assay_items:
                    fh.write(item.source_id + "\n")
            print(f"Saved assay IDs  → {args.save_assay_ids}")

    print(f"\nComputing coverage (covered_thr={args.covered_thr}) …")
    d_max = initial_coverage_scores(train_vecs, assay_vecs)

    covered_mask   = d_max >= args.covered_thr
    uncovered_mask = ~covered_mask
    n_uncovered    = int(uncovered_mask.sum())
    n_covered      = int(covered_mask.sum())

    print(f"  covered   : {n_covered:,}  ({n_covered/len(d_max):.1%})")
    print(f"  uncovered : {n_uncovered:,}  ({n_uncovered/len(d_max):.1%})")

    if n_uncovered == 0:
        raise SystemExit("All assays are already covered — nothing to cluster.")

    uncovered_idx  = np.where(uncovered_mask)[0]
    uncovered_vecs = assay_vecs[uncovered_idx]

    if args.load_labels and args.load_labels.exists():
        print(f"Loading cached cluster labels: {args.load_labels} (skipping UMAP+HDBSCAN)")
        labels = np.load(str(args.load_labels)).astype(np.int32)
        assert len(labels) == n_uncovered, (
            f"Labels cache size mismatch: {len(labels)} vs {n_uncovered} uncovered. "
            "Delete cache and rerun."
        )
    else:
        if args.umap_cache and args.umap_cache.exists():
            print(f"Loading cached UMAP reduction: {args.umap_cache}")
            reduced = np.load(str(args.umap_cache)).astype(np.float32)
            assert len(reduced) == n_uncovered, "UMAP cache size mismatch — delete and rerun."
        else:
            reduced = reduce_umap(
                uncovered_vecs,
                n_components=args.umap_components,
                n_neighbors=args.umap_neighbors,
            )
            if args.umap_cache:
                args.umap_cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(args.umap_cache), reduced)
                print(f"Saved UMAP cache → {args.umap_cache}")

        labels = run_hdbscan(
            reduced,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_method=args.cluster_method,
        )
        if args.save_labels:
            args.save_labels.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(args.save_labels), labels)
            print(f"Saved cluster labels → {args.save_labels}")

    cluster_ids    = sorted(set(labels[labels >= 0]))
    cluster_sizes  = {cid: int(np.sum(labels == cid)) for cid in cluster_ids}
    noise_count    = int(np.sum(labels == -1))

    print(f"\nCluster size distribution:")
    sizes = sorted(cluster_sizes.values(), reverse=True)
    print(f"  top-10 sizes : {sizes[:10]}")
    print(f"  median size  : {int(np.median(sizes))}")
    print(f"  noise points : {noise_count:,}")

    medoids = find_medoids(uncovered_vecs, labels, n_medoids=args.n_examples)

    tokenizer_path = args.tokenizer or args.model
    print(f"\nLoading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    sorted_clusters = sorted(cluster_ids, key=lambda c: cluster_sizes[c], reverse=True)

    if args.top_clusters and args.top_clusters > 0:
        sorted_clusters = sorted_clusters[: args.top_clusters]
        total_covered_by_selection = sum(cluster_sizes[c] for c in sorted_clusters)
        print(
            f"\nTop-{args.top_clusters} clusters selected "
            f"→ covers {total_covered_by_selection:,} assays "
            f"({total_covered_by_selection / n_uncovered:.1%} of uncovered)"
        )

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    records    = []
    summary    = []

    for rank, cid in enumerate(tqdm(sorted_clusters, desc="Building records")):
        size = cluster_sizes[cid]
        for local_idx in medoids[cid]:
            full_idx = int(uncovered_idx[local_idx])
            item     = assay_items[full_idx]
            sim      = float(d_max[full_idx])

            tok = tokenize_for_ner(item.text, tokenizer, args.max_length)

            record = {
                "text":           item.text,
                "tokens":         tok["tokens"],
                "tags":           tok["tags"],
                "input_ids":      tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "meta": {
                    **item.meta,
                    "cluster_id":      int(cid),
                    "cluster_size":    size,
                    "cluster_rank":    rank,
                    "sim_to_train":    round(sim, 4),
                    "source_assay_id": item.source_id,
                    "annotation_status": "pending",
                },
            }
            records.append(record)

            summary.append({
                "cluster_rank":    rank,
                "cluster_id":      int(cid),
                "cluster_size":    size,
                "assay_id":        item.source_id,
                "sim_to_train":    round(sim, 4),
                "text_preview":    item.text[:200],
            })

    json_path = out_prefix.with_name(out_prefix.name + ".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved annotation JSON  → {json_path}  ({len(records)} records)")

    tsv_path = out_prefix.with_name(out_prefix.name + "_summary.tsv")
    with tsv_path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(summary[0].keys()) + "\n")
        for row in summary:
            fh.write("\t".join(str(v) for v in row.values()) + "\n")
    print(f"Saved cluster summary  → {tsv_path}")

    stats_path = out_prefix.with_name(out_prefix.name + "_cluster_stats.json")
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "covered_thr":       args.covered_thr,
                "n_total_assays":    len(assay_items),
                "n_uncovered":       n_uncovered,
                "n_clusters":        len(cluster_ids),
                "n_noise":           noise_count,
                "noise_rate":        round(noise_count / n_uncovered, 4),
                "n_examples_per_cluster": args.n_examples,
                "n_records_output":  len(records),
                "cluster_sizes":     {
                    str(cid): cluster_sizes[cid]
                    for cid in sorted_clusters
                },
                "hdbscan": {
                    "min_cluster_size":      args.min_cluster_size,
                    "min_samples":           args.min_samples,
                    "cluster_selection":     args.cluster_method,
                },
                "umap": {
                    "n_components": args.umap_components,
                    "n_neighbors":  args.umap_neighbors,
                },
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved cluster stats    → {stats_path}")

    print(f"""
──────────────────────────────────────────────────
Next steps:
  1. Open {json_path.name}
  2. Annotate 'tags' fields (B-TAR / I-TAR / B-SUB / I-SUB / O)
     Start with cluster_rank=0 (largest uncovered type first)
  3. Flip annotation_status: "pending" → "done"
  4. Concatenate with your existing train JSON:
       python3 -c "
       import json
       old = json.load(open('data/biomedbert_ready_data_dedup.json'))
       new = [r for r in json.load(open('{json_path}'))
              if r['meta']['annotation_status'] == 'done']
       json.dump(old + new, open('data/biomedbert_ready_data_dedup_v2.json','w'),
                 ensure_ascii=False, indent=2)
       print(f'Train size: {{len(old)}} → {{len(old+new)}}')
       "
  5. Retrain BERT NER
──────────────────────────────────────────────────
""")

if __name__ == "__main__":
    main()
