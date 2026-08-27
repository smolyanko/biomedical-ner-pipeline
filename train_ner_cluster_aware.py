from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2, IOBES
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from nervaluate import Evaluator as NervaluateEvaluator
except Exception:
    try:
        from nervaluate.evaluator import Evaluator as NervaluateEvaluator
    except Exception:
        NervaluateEvaluator = None

try:
    import optuna
except Exception:
    optuna = None

LABEL_PAD_ID = -100
DEFAULT_OPTUNA_TRIALS = 30
DEFAULT_INPUT_SCHEME = "BIO"
DEFAULT_OUTPUT_SCHEME = "BIOES"
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}

NOISE_CLUSTER_ID = -1

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text.lower().strip())

def _safe_text(ex: Dict) -> str:
    text = ex.get("text")
    if isinstance(text, str) and text.strip():
        return text
    toks = ex.get("tokens") or ex.get("words") or []
    return " ".join(map(str, toks))

def split_tag(tag: str) -> Tuple[str, Optional[str]]:
    if tag == "O":
        return "O", None
    if "-" not in tag:
        return tag, None
    return tag.split("-", 1)

def bio_to_spans(tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    start = None
    ent = None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, typ = split_tag(tag)
        if tag == "O":
            if start is not None:
                spans.append((start, i - 1, ent))
            start = ent = None
        elif prefix == "B":
            if start is not None:
                spans.append((start, i - 1, ent))
            start, ent = i, typ
        elif prefix == "I":
            if start is None or ent != typ:
                if start is not None:
                    spans.append((start, i - 1, ent))
                start, ent = i, typ
        elif prefix in {"S", "E"}:
            if start is not None:
                spans.append((start, i - 1, ent))
            spans.append((i, i, typ))
            start = ent = None
    return spans

def bio_to_bioes(tags: Sequence[str]) -> List[str]:
    out = ["O"] * len(tags)
    for s, e, typ in bio_to_spans(tags):
        if s == e:
            out[s] = f"S-{typ}"
        else:
            out[s] = f"B-{typ}"
            for i in range(s + 1, e):
                out[i] = f"I-{typ}"
            out[e] = f"E-{typ}"
    return out

def bioes_to_bio(tags: Sequence[str]) -> List[str]:
    out = []
    for tag in tags:
        p, t = split_tag(tag)
        if p in ("B", "S"):
            out.append(f"B-{t}" if t else "O")
        elif p in ("I", "E"):
            out.append(f"I-{t}" if t else "O")
        else:
            out.append("O")
    return out

def normalize_tags(tags: Sequence[str], in_scheme: str, out_scheme: str) -> List[str]:
    i, o = in_scheme.upper(), out_scheme.upper()
    if i == o:
        return list(tags)
    if i == "BIO" and o == "BIOES":
        return bio_to_bioes(tags)
    if i == "BIOES" and o == "BIO":
        return bioes_to_bio(tags)
    raise ValueError(f"Unsupported tag conversion: {i} -> {o}")

def tags_to_spans(tags: List[str]) -> List[Dict]:
    spans: List[Dict] = []
    cur = None
    for i, tag in enumerate(tags):
        p, lbl = split_tag(tag)
        if p in ("B", "S"):
            if cur:
                spans.append(cur)
            cur = {"label": lbl, "start": i, "end": i}
            if p == "S":
                spans.append(cur)
                cur = None
        elif p in ("I", "E") and cur:
            if lbl == cur["label"]:
                cur["end"] = i
            else:
                spans.append(cur)
                cur = {"label": lbl, "start": i, "end": i}
            if p == "E":
                spans.append(cur)
                cur = None
        elif tag == "O" and cur:
            spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return spans

def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of examples")
    return data

def _reconstruct_words_and_tags(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str]]:
    words: List[str] = []
    word_tags: List[str] = []
    current_word = None
    current_tag = None

    for tok, tag in zip(tokens, tags):
        if tok in SPECIAL_TOKENS:
            if current_word is not None:
                words.append(current_word)
                word_tags.append(current_tag)
                current_word = None
                current_tag = None
            continue
        if tok.startswith("##"):
            if current_word is None:
                current_word = tok[2:]
                current_tag = tag
            else:
                current_word += tok[2:]
        else:
            if current_word is not None:
                words.append(current_word)
                word_tags.append(current_tag)
            current_word = tok
            current_tag = tag

    if current_word is not None:
        words.append(current_word)
        word_tags.append(current_tag)

    return words, word_tags

def clean_examples(examples: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for ex in examples:
        tokens = ex.get("tokens") or ex.get("words")
        tags = ex.get("tags")
        if not tokens or not tags or len(tokens) != len(tags):
            continue
        out.append({
            "text": ex.get("text", ""),
            "tokens": list(tokens),
            "tags": list(tags),
            "meta": dict(ex.get("meta") or {}),
        })
    return out

def extract_entity_types(examples: Sequence[Dict]) -> List[str]:
    types = set()
    for ex in examples:
        for tag in ex["tags"]:
            _, t = split_tag(tag)
            if t:
                types.add(t)
    return sorted(types)

def build_label_list(entity_types: Sequence[str], scheme: str = "BIOES") -> List[str]:
    labels = ["O"]
    if scheme.upper() == "BIO":
        for t in entity_types:
            labels += [f"B-{t}", f"I-{t}"]
    else:
        for t in entity_types:
            labels += [f"B-{t}", f"I-{t}", f"E-{t}", f"S-{t}"]
    return labels

def _encode_for_clustering(
    texts: List[str],
    model_name: str,
    device: Optional[str],
    batch_size: int,
    logger: logging.Logger,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit("sentence-transformers is required for clustering") from e

    kwargs = {"device": device} if device else {}
    logger.info(f"[cluster] loading embedder: {model_name}")
    model = SentenceTransformer(model_name, **kwargs)
    logger.info(f"[cluster] encoding {len(texts):,} texts")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)

def _reduce_umap(
    vecs: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    logger: logging.Logger,
    cache_path: Optional[Path] = None,
    random_state: int = 42,
) -> np.ndarray:
    if cache_path and cache_path.exists():
        logger.info(f"[cluster] loading cached UMAP reduction: {cache_path}")
        reduced = np.load(str(cache_path)).astype(np.float32)
        if len(reduced) != len(vecs):
            raise ValueError(
                f"UMAP cache size mismatch: {len(reduced)} vs {len(vecs)} — delete cache and rerun"
            )
        return reduced

    try:
        import umap as umap_lib
    except ImportError as e:
        raise SystemExit("pip install umap-learn is required for HDBSCAN-based split") from e

    logger.info(
        f"[cluster] UMAP: {vecs.shape} -> ({len(vecs)}, {n_components}) "
        f"n_neighbors={n_neighbors} min_dist={min_dist}"
    )
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        low_memory=True,
        verbose=True,
    )
    reduced = reducer.fit_transform(vecs).astype(np.float32)

    if cache_path:
        np.save(str(cache_path), reduced)
        logger.info(f"[cluster] UMAP reduction cached to {cache_path}")

    return reduced

def cluster_from_embeddings(
    examples: List[Dict],
    embed_model: str,
    embed_device: Optional[str],
    embed_batch: int,
    vecs_cache: Optional[Path],
    logger: logging.Logger,
    umap_components: int = 50,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    umap_cache: Optional[Path] = None,
    min_cluster_size: int = 30,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",
) -> List[Dict]:
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        backend = "sklearn"
    except ImportError:
        try:
            import hdbscan as hdbscan_lib
            backend = "hdbscan_pkg"
        except ImportError as e:
            raise SystemExit(
                "Need either scikit-learn >= 1.3 (sklearn.cluster.HDBSCAN) "
                "or the standalone 'hdbscan' package."
            ) from e

    texts = [_normalize(_safe_text(ex)) for ex in examples]

    if vecs_cache and vecs_cache.exists():
        logger.info(f"[cluster] loading cached embeddings: {vecs_cache}")
        vecs = np.load(str(vecs_cache)).astype(np.float32)
        if len(vecs) != len(examples):
            raise ValueError(f"embedding cache size mismatch: {len(vecs)} vs {len(examples)}")
    else:
        vecs = _encode_for_clustering(texts, embed_model, embed_device, embed_batch, logger)
        if vecs_cache:
            np.save(str(vecs_cache), vecs)
            logger.info(f"[cluster] embeddings cached to {vecs_cache}")

    reduced = _reduce_umap(
        vecs,
        n_components=umap_components,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        logger=logger,
        cache_path=umap_cache,
    )

    logger.info(
        f"[cluster] HDBSCAN (backend={backend}): "
        f"min_cluster_size={min_cluster_size} min_samples={min_samples} "
        f"cluster_selection_method={cluster_selection_method}"
    )
    if backend == "sklearn":
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",
            n_jobs=-1,
        )
    else:
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",
            core_dist_n_jobs=-1,
            prediction_data=True,
        )
    labels = clusterer.fit_predict(reduced).astype(int)

    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    n_noise = int(np.sum(labels == -1))
    logger.info(
        f"[cluster] HDBSCAN found {n_clusters} clusters, "
        f"{n_noise} noise points ({n_noise / len(labels):.1%})"
    )

    if n_clusters > 0:
        sizes = Counter(labels[labels >= 0].tolist())
        logger.info(
            "[cluster] cluster size stats (excluding noise): min=%s max=%s median=%s",
            min(sizes.values()), max(sizes.values()), int(np.median(list(sizes.values()))),
        )

    result = []
    for ex, cid in zip(examples, labels):
        cid = int(cid)
        new_ex = dict(ex)
        new_ex["meta"] = dict(ex.get("meta") or {})
        new_ex["meta"]["cluster"] = cid if cid >= 0 else NOISE_CLUSTER_ID
        result.append(new_ex)
    return result

def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

def cluster_split(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    _validate_ratios(train_ratio, val_ratio)
    rng = random.Random(42)

    by_cluster: Dict[int, List[Dict]] = defaultdict(list)
    for ex in examples:
        cid = ex.get("meta", {}).get("cluster", NOISE_CLUSTER_ID)
        by_cluster[cid].append(ex)

    noise_examples = by_cluster.pop(NOISE_CLUSTER_ID, [])

    cluster_ids = sorted(by_cluster.keys())
    rng.shuffle(cluster_ids)

    n_c = len(cluster_ids)
    n_train_c = max(1, int(n_c * train_ratio))
    n_val_c   = max(1, int(n_c * val_ratio))
    n_test_c  = n_c - n_train_c - n_val_c

    if n_test_c < 1:
        n_train_c -= 1
        n_test_c = 1
    if n_val_c < 1:
        n_val_c = 1
        n_train_c = max(1, n_train_c - 1)

    train_clusters = set(cluster_ids[:n_train_c])
    val_clusters   = set(cluster_ids[n_train_c : n_train_c + n_val_c])

    train_all, val_all, test_all = [], [], []
    for cid, exs in by_cluster.items():
        if cid in train_clusters:
            train_all.extend(exs)
        elif cid in val_clusters:
            val_all.extend(exs)
        else:
            test_all.extend(exs)

    train_all.extend(noise_examples)

    if logger:
        n_test_c_actual = n_c - n_train_c - n_val_c
        logger.info(
            f"Cluster split (cluster-aware, HDBSCAN): "
            f"train={len(train_all)} ({n_train_c} clusters + {len(noise_examples)} noise) | "
            f"val={len(val_all)} ({n_val_c} clusters) | "
            f"test={len(test_all)} ({n_test_c_actual} clusters)"
        )
    return train_all, val_all, test_all

def _shuffle_list(items: List[Dict]) -> List[Dict]:
    items = list(items)
    random.Random(42).shuffle(items)
    return items

def load_and_split(
    data_path: Path,
    train_ratio: float,
    val_ratio: float,
    logger: logging.Logger,
    embed_model: str = "kamalkraj/BioSimCSE-BioLinkBERT-BASE",
    embed_device: Optional[str] = None,
    embed_batch: int = 64,
    vecs_cache: Optional[Path] = None,
    umap_components: int = 50,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    umap_cache: Optional[Path] = None,
    min_cluster_size: int = 30,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    logger.info(f"Loading dataset: {data_path}")
    raw = load_json(data_path)
    clean = clean_examples(raw)
    logger.info(f"Clean examples: {len(clean)}")

    clustered = cluster_from_embeddings(
        clean,
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
    tr, va, te = cluster_split(clustered, train_ratio, val_ratio, logger=logger)
    tr = _shuffle_list(tr)
    va = _shuffle_list(va)
    te = _shuffle_list(te)
    logger.info(f"Final split: train={len(tr)} | dev={len(va)} | test={len(te)}")
    return tr, va, te

def _encode_words(
    words: List[str],
    tokenizer,
    max_length: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors=None,
    )
    input_ids = list(enc["input_ids"])
    attention_mask = list(enc["attention_mask"])

    if not hasattr(enc, "word_ids"):
        raise RuntimeError(
            "Tokenizer must be fast (support word_ids). Use a fast tokenizer."
        )
    word_ids = enc.word_ids()

    word_starts: List[int] = []
    kept_word_ids: List[int] = []
    seen = set()
    for idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if w_id not in seen:
            seen.add(w_id)
            word_starts.append(idx)
            kept_word_ids.append(w_id)

    return input_ids, attention_mask, word_starts, kept_word_ids

def retokenize_example(
    ex: Dict,
    tokenizer,
    label2id: Dict[str, int],
    input_scheme: str,
    output_scheme: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    tokens = ex["tokens"]
    tags = normalize_tags(ex["tags"], input_scheme, output_scheme)
    words, word_tags = _reconstruct_words_and_tags(tokens, tags)

    input_ids, attention_mask, word_starts, kept_word_ids = _encode_words(
        words, tokenizer, max_length=max_length
    )

    word_labels = [label2id[word_tags[w_id]] for w_id in kept_word_ids]
    word_mask = [1] * len(word_labels)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "word_starts": torch.tensor(word_starts, dtype=torch.long),
        "word_mask": torch.tensor(word_mask, dtype=torch.long),
        "labels": torch.tensor(word_labels, dtype=torch.long),
    }

def prepare_items_retokenize(
    examples: Sequence[Dict],
    tokenizer,
    label2id: Dict[str, int],
    input_scheme: str,
    output_scheme: str,
    max_length: int,
) -> List[Dict[str, torch.Tensor]]:
    return [
        retokenize_example(ex, tokenizer, label2id, input_scheme, output_scheme, max_length)
        for ex in examples
    ]

class NERDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

def collate_fn_factory(pad_token_id: int):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [x["input_ids"] for x in batch], batch_first=True, padding_value=pad_token_id
            ),
            "attention_mask": pad_sequence(
                [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
            ),
            "word_starts": pad_sequence(
                [x["word_starts"] for x in batch], batch_first=True, padding_value=-1
            ),
            "word_mask": pad_sequence(
                [x["word_mask"] for x in batch], batch_first=True, padding_value=0
            ),
            "labels": pad_sequence(
                [x["labels"] for x in batch], batch_first=True, padding_value=LABEL_PAD_ID
            ),
        }
    return collate

def compute_class_weights(
    examples: Sequence[Dict],
    label2id: Dict[str, int],
    input_scheme: str,
    output_scheme: str,
    device: torch.device,
    o_weight: float = 0.1,
) -> torch.Tensor:
    counts = Counter()
    for ex in examples:
        tags = normalize_tags(ex["tags"], input_scheme, output_scheme)
        counts.update(tags)

    weights = torch.ones(len(label2id), dtype=torch.float32)
    for label, idx in label2id.items():
        if label == "O":
            weights[idx] = o_weight
        else:
            c = counts.get(label, 0)
            weights[idx] = 1.0 / max(math.sqrt(c), 1e-6) if c > 0 else 1.0

    entity_mask = torch.tensor([lbl != "O" for lbl in label2id.keys()], dtype=torch.bool)
    if entity_mask.any():
        weights[entity_mask] /= weights[entity_mask].mean().clamp_min(1e-6)

    return weights.to(device)

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    ignore_index: int = LABEL_PAD_ID,
) -> torch.Tensor:
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0

    logits_v = logits[valid]
    labels_v = labels[valid]

    ce = F.cross_entropy(
        logits_v, labels_v,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    with torch.no_grad():
        log_pt = F.log_softmax(logits_v, dim=-1).gather(1, labels_v.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

    return (((1.0 - pt) ** gamma) * ce).mean()

def setup_logging(save_dir: Path) -> logging.Logger:
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in [
        logging.FileHandler(save_dir / "run.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

def _gather_word_representations(
    hidden_states: torch.Tensor,
    word_starts: torch.Tensor,
    word_mask: torch.Tensor,
) -> torch.Tensor:
    _, _, hidden = hidden_states.shape
    starts = word_starts.clamp(min=0)
    gathered = hidden_states.gather(1, starts.unsqueeze(-1).expand(-1, -1, hidden))
    return gathered * word_mask.unsqueeze(-1).to(hidden_states.dtype)

class BertTokenClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=False)
        if gradient_checkpointing and hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()
            self.bert.config.use_cache = False
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

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

        ce_loss = F.cross_entropy(
            logits[valid], labels[valid],
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        if self.focal_weight > 0:
            fl = focal_loss(logits, labels, self.class_weights, self.focal_gamma, self.label_smoothing)
            loss = ce_loss + self.focal_weight * fl
        else:
            loss = ce_loss
        return {"loss": loss, "predictions": preds, "mask": word_mask.bool()}

class BertCRFForNER(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        focal_weight: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        use_layer_pooling: bool = True,
        n_last_layers: int = 4,
        mlp_hidden: int = 256,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name, output_hidden_states=use_layer_pooling
        )
        if gradient_checkpointing and hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()
            self.bert.config.use_cache = False

        self.use_layer_pooling = use_layer_pooling
        self.n_last_layers = n_last_layers
        h = self.bert.config.hidden_size

        if use_layer_pooling:
            self.layer_weights = nn.Parameter(torch.zeros(n_last_layers))
        else:
            self.layer_weights = None

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(h, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, h),
        )
        self.classifier = nn.Linear(h, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def _pool(self, out) -> torch.Tensor:
        if self.use_layer_pooling and out.hidden_states is not None:
            n = min(self.n_last_layers, len(out.hidden_states))
            if n <= 1:
                return out.hidden_states[-1]
            weights = torch.softmax(self.layer_weights[-n:], dim=0)
            return sum(w * h for w, h in zip(weights, out.hidden_states[-n:]))
        return out.last_hidden_state

    def forward(self, input_ids, attention_mask, word_starts, word_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = _gather_word_representations(self._pool(out), word_starts, word_mask)
        seq = seq + self.mlp(self.dropout(seq))
        emissions = self.classifier(seq)
        mask = word_mask.bool()

        preds = self.crf.decode(emissions, mask=mask)

        if labels is None:
            return {"predictions": preds, "mask": mask}

        labels_crf = labels.clone()
        labels_crf[labels_crf == LABEL_PAD_ID] = 0
        crf_loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")

        if self.focal_weight > 0:
            fl = focal_loss(
                emissions, labels, self.class_weights,
                gamma=self.focal_gamma, label_smoothing=self.label_smoothing,
            )
            loss = crf_loss + self.focal_weight * fl
        else:
            loss = crf_loss

        return {"loss": loss, "predictions": preds, "mask": mask}

class BertBiLSTMCRFForNER(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        class_weights: Optional[torch.Tensor] = None,
        focal_weight: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=False)
        if gradient_checkpointing and hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()
            self.bert.config.use_cache = False
        h = self.bert.config.hidden_size
        self.bilstm = nn.LSTM(
            h,
            lstm_hidden // 2,
            lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask, word_starts, word_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = _gather_word_representations(out.last_hidden_state, word_starts, word_mask)
        seq, _ = self.bilstm(seq)
        emissions = self.classifier(self.dropout(seq))
        mask = word_mask.bool()
        preds = self.crf.decode(emissions, mask=mask)

        if labels is None:
            return {"predictions": preds, "mask": mask}

        labels_crf = labels.clone()
        labels_crf[labels_crf == LABEL_PAD_ID] = 0
        crf_loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")

        if self.focal_weight > 0:
            fl = focal_loss(
                emissions, labels, self.class_weights,
                gamma=self.focal_gamma, label_smoothing=self.label_smoothing,
            )
            loss = crf_loss + self.focal_weight * fl
        else:
            loss = crf_loss
        return {"loss": loss, "predictions": preds, "mask": mask}

class BertLoRAForNER(nn.Module):

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        classifier_type: str = "crf",
        backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        class_weights: Optional[torch.Tensor] = None,
        focal_weight: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as e:
            raise SystemExit("peft is required for LoRA") from e

        base = AutoModel.from_pretrained(model_name, output_hidden_states=False)

        if backbone_state_dict is not None:
            base.load_state_dict(backbone_state_dict, strict=False)

        target_modules = _infer_lora_targets(base)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.bert = get_peft_model(base, lora_cfg)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base.config.hidden_size, num_labels)
        self.classifier_type = classifier_type
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if classifier_type == "crf":
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.crf = None
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask, word_starts, word_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = _gather_word_representations(out.last_hidden_state, word_starts, word_mask)
        emissions = self.classifier(self.dropout(seq))
        mask = word_mask.bool()

        if self.crf is None:
            preds = emissions.argmax(-1).tolist()
            if labels is None:
                return {"predictions": preds, "mask": mask}
            valid = labels != LABEL_PAD_ID
            if not valid.any():
                return {"loss": emissions.sum() * 0.0, "predictions": preds, "mask": mask}
            ce_loss = F.cross_entropy(
                emissions[valid], labels[valid],
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            if self.focal_weight > 0:
                fl = focal_loss(emissions, labels, self.class_weights, self.focal_gamma, self.label_smoothing)
                loss = ce_loss + self.focal_weight * fl
            else:
                loss = ce_loss
            return {"loss": loss, "predictions": preds, "mask": mask}

        preds = self.crf.decode(emissions, mask=mask)
        if labels is None:
            return {"predictions": preds, "mask": mask}
        labels_crf = labels.clone()
        labels_crf[labels_crf == LABEL_PAD_ID] = 0
        crf_loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")
        if self.focal_weight > 0:
            fl = focal_loss(emissions, labels, self.class_weights, self.focal_gamma, self.label_smoothing)
            loss = crf_loss + self.focal_weight * fl
        else:
            loss = crf_loss
        return {"loss": loss, "predictions": preds, "mask": mask}

    def print_trainable_parameters(self, logger: Optional[logging.Logger] = None) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        msg = f"LoRA trainable params: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)"
        if logger:
            logger.info(msg)
        return msg

def _infer_lora_targets(model: nn.Module) -> List[str]:
    names = set()
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        leaf = n.split(".")[-1].lower()
        if leaf in ("query", "value"):
            names.add(n.split(".")[-1])
        if leaf in ("q_proj", "v_proj"):
            names.add(n.split(".")[-1])
    return sorted(names) if names else ["query", "value"]

def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "LayerNorm.weight" in name
            or "layer_norm.weight" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]

    return torch.optim.AdamW(param_groups, lr=lr)

def get_amp(use_amp: bool):
    if not use_amp or not torch.cuda.is_available():
        return None, None, False
    bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = None if bf16 else torch.cuda.amp.GradScaler()
    return dtype, scaler, bf16

def decode_predictions(predictions, labels, mask, id2label):
    true_tags, pred_tags = [], []
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    for pred, gold, m in zip(predictions, labels, mask):
        n = int(np.sum(m))
        seq_len = min(len(pred), n)
        row_pred, row_true = [], []
        for p, g in zip(pred[:seq_len], gold[:seq_len]):
            if int(g) == LABEL_PAD_ID:
                continue
            row_pred.append(id2label[int(p)])
            row_true.append(id2label[int(g)])
        if row_true:
            true_tags.append(row_true)
            pred_tags.append(row_pred)
    return true_tags, pred_tags

@torch.no_grad()
def evaluate(model, loader, id2label, scheme, device):
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_starts=batch["word_starts"],
            word_mask=batch["word_mask"],
        )
        t, p = decode_predictions(out["predictions"], batch["labels"], out["mask"], id2label)
        all_true.extend(t)
        all_pred.extend(p)

    seq_scheme = IOBES if scheme.upper() == "BIOES" else IOB2
    return {
        "precision": precision_score(all_true, all_pred, mode="strict", scheme=seq_scheme) if all_true else 0.0,
        "recall": recall_score(all_true, all_pred, mode="strict", scheme=seq_scheme) if all_true else 0.0,
        "f1": f1_score(all_true, all_pred, mode="strict", scheme=seq_scheme) if all_true else 0.0,
        "true_tags": all_true,
        "pred_tags": all_pred,
    }

@torch.no_grad()
def final_evaluate(model, loader, id2label, device, entity_types) -> Dict[str, float]:
    if NervaluateEvaluator is None:
        return {}

    out = evaluate(model, loader, id2label, "BIOES", device)
    true_sp = [tags_to_spans(ts) for ts in out["true_tags"]]
    pred_sp = [tags_to_spans(ps) for ps in out["pred_tags"]]

    try:
        res = NervaluateEvaluator(true_sp, pred_sp, tags=entity_types).evaluate()
    except TypeError:
        res = NervaluateEvaluator(true_sp, pred_sp, tags=entity_types, loader="list").evaluate()

    if isinstance(res, (tuple, list)) and res:
        res = res[0]

    out_metrics: Dict[str, float] = {}
    for key in ("strict", "partial", "ent_type", "exact"):
        if key in res and isinstance(res[key], dict):
            for m in ("precision", "recall", "f1"):
                out_metrics[f"{key}_{m}"] = float(res[key].get(m, 0.0))
    return out_metrics

def train_epoch(model, loader, optimizer, scheduler, device, use_amp, amp_dtype, scaler, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pending = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_amp and torch.cuda.is_available():
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(**batch)["loss"] / grad_accum
        else:
            loss = model(**batch)["loss"] / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        pending += 1
        if pending == grad_accum:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

        total_loss += float(loss.item()) * grad_accum

    if pending > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, len(loader))

@dataclass
class TrainConfig:
    architecture: str
    model_name: str
    input_scheme: str
    scheme: str
    max_length: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    dropout: float
    patience: int = 4
    grad_accum_steps: int = 1
    num_workers: int = 4
    use_amp: bool = True
    focal_weight: float = 0.0
    focal_gamma: float = 2.0
    o_weight: float = 0.1
    label_smoothing: float = 0.0
    use_layer_pooling: bool = False
    n_last_layers: int = 0
    mlp_hidden: int = 0
    gradient_checkpointing: bool = False
    lstm_hidden: int = 256
    lstm_layers: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_classifier_type: str = "crf"

ARCHITECTURES: Dict[str, Dict] = {
    "bert_linear": {
        "use_layer_pooling": False,
        "focal_weight": 0.0,
        "lstm_hidden": 0,
        "lstm_layers": 0,
    },
    "bert_crf": {
        "use_layer_pooling": True,
        "focal_weight": 0.1,
        "n_last_layers": 4,
        "mlp_hidden": 256,
        "lstm_hidden": 0,
        "lstm_layers": 0,
    },
    "bert_bilstm_crf": {
        "use_layer_pooling": False,
        "focal_weight": 0.1,
        "lstm_hidden": 256,
        "lstm_layers": 1,
    },
    "bert_lora": {
        "use_layer_pooling": False,
        "focal_weight": 0.1,
        "lstm_hidden": 0,
        "lstm_layers": 0,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_classifier_type": "crf",
    },
}

DEFAULT_ARCHITECTURES = ["bert_linear", "bert_crf", "bert_bilstm_crf", "bert_lora"]

def build_model(
    config: TrainConfig,
    num_labels: int,
    class_weights: torch.Tensor,
    backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    arch = config.architecture
    if arch == "bert_linear":
        return BertTokenClassifier(
            config.model_name, num_labels, config.dropout,
            class_weights=class_weights,
            focal_weight=config.focal_weight,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            gradient_checkpointing=config.gradient_checkpointing,
        )
    if arch == "bert_crf":
        return BertCRFForNER(
            config.model_name, num_labels, config.dropout,
            class_weights=class_weights,
            focal_weight=config.focal_weight,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            use_layer_pooling=config.use_layer_pooling,
            n_last_layers=config.n_last_layers,
            mlp_hidden=config.mlp_hidden,
            gradient_checkpointing=config.gradient_checkpointing,
        )
    if arch == "bert_bilstm_crf":
        return BertBiLSTMCRFForNER(
            config.model_name, num_labels, config.dropout,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            class_weights=class_weights,
            focal_weight=config.focal_weight,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            gradient_checkpointing=config.gradient_checkpointing,
        )
    if arch == "bert_lora":
        return BertLoRAForNER(
            config.model_name, num_labels, config.dropout,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            classifier_type=config.lora_classifier_type,
            backbone_state_dict=backbone_state_dict,
            class_weights=class_weights,
            focal_weight=config.focal_weight,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
    raise ValueError(f"Unknown architecture: {arch}")

def architecture_trial_space(trial, architecture: str) -> Dict:
    params: Dict = {
        "batch_size": trial.suggest_categorical("batch_size", [2, 4, 6, 8]),
        "lr": trial.suggest_float("lr", 1e-5, 8e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.1, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.03, 0.2),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "o_weight": trial.suggest_float("o_weight", 0.05, 0.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "patience": trial.suggest_categorical("patience", [2, 3, 4, 5]),
    }

    if architecture == "bert_crf":
        params.update({
            "use_layer_pooling": trial.suggest_categorical("use_layer_pooling", [True, False]),
            "focal_weight": trial.suggest_float("focal_weight", 0.0, 0.3),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.5, 4.0),
            "n_last_layers": trial.suggest_int("n_last_layers", 2, 6),
            "mlp_hidden": trial.suggest_categorical("mlp_hidden", [128, 256, 384, 512]),
        })
    elif architecture == "bert_bilstm_crf":
        params.update({
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [128, 256, 384]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
            "focal_weight": trial.suggest_float("focal_weight", 0.0, 0.3),
        })
    elif architecture == "bert_lora":
        params.update({
            "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16, 32]),
            "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32, 64]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
            "focal_weight": trial.suggest_float("focal_weight", 0.0, 0.3),
        })
    return params

def train_one_run(
    train_examples: Sequence[Dict],
    dev_examples: Sequence[Dict],
    test_examples: Sequence[Dict],
    config: TrainConfig,
    device: torch.device,
    logger: logging.Logger,
    trial=None,
    backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    seed_everything(42)

    train_norm = [
        {**ex, "tags": normalize_tags(ex["tags"], config.input_scheme, config.scheme)}
        for ex in train_examples
    ]
    dev_norm = [
        {**ex, "tags": normalize_tags(ex["tags"], config.input_scheme, config.scheme)}
        for ex in dev_examples
    ]
    test_norm = [
        {**ex, "tags": normalize_tags(ex["tags"], config.input_scheme, config.scheme)}
        for ex in test_examples
    ]

    entity_types = extract_entity_types(train_norm + dev_norm + test_norm)
    labels = build_label_list(entity_types, config.scheme)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    logger.info(f"[{config.architecture}] labels={labels}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for word alignment")

    collate = collate_fn_factory(tokenizer.pad_token_id or 0)
    loader_kwargs = dict(
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"[{config.architecture}] tokenizing with {config.model_name}")

    train_ds = NERDataset(
        prepare_items_retokenize(train_norm, tokenizer, label2id, config.scheme, config.scheme, config.max_length)
    )
    dev_ds = NERDataset(
        prepare_items_retokenize(dev_norm, tokenizer, label2id, config.scheme, config.scheme, config.max_length)
    )
    test_ds = NERDataset(
        prepare_items_retokenize(test_norm, tokenizer, label2id, config.scheme, config.scheme, config.max_length)
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    class_weights = compute_class_weights(
        train_norm, label2id, config.scheme, config.scheme, device, config.o_weight
    )

    model = build_model(config, len(labels), class_weights, backbone_state_dict=backbone_state_dict).to(device)

    if hasattr(model, "print_trainable_parameters"):
        try:
            model.print_trainable_parameters(logger)
        except Exception:
            pass

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[{config.architecture}] model={type(model).__name__} "
        f"total={n_total:,} trainable={n_train:,} ({100*n_train/max(n_total,1):.2f}%)"
    )

    optimizer = build_optimizer(model, config.lr, config.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / max(1, config.grad_accum_steps)) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    amp_dtype, scaler, bf16 = get_amp(config.use_amp)
    if config.use_amp:
        logger.info(f"[{config.architecture}] AMP: {'bf16' if bf16 else 'fp16'}")

    best_state = None
    best_dev_score = -1e9
    best_epoch = -1
    best_dev_metrics = None
    bad_epochs = 0

    for epoch in range(config.epochs):
        loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            config.use_amp, amp_dtype, scaler, config.grad_accum_steps,
        )
        dev_m = evaluate(model, dev_loader, id2label, config.scheme, device)
        logger.info(
            f"[{config.architecture}] epoch={epoch+1}/{config.epochs} loss={loss:.4f} "
            f"dev_f1={dev_m['f1']:.4f} p={dev_m['precision']:.4f} r={dev_m['recall']:.4f}"
        )

        score = dev_m["f1"]
        if score > best_dev_score:
            best_dev_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_dev_metrics = dict(dev_m)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                logger.info(f"[{config.architecture}] early stopping at epoch {epoch+1}")
                break

        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_seqeval = evaluate(model, test_loader, id2label, config.scheme, device)
    test_nervaluate = final_evaluate(model, test_loader, id2label, device, entity_types)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label2id": label2id,
        "id2label": id2label,
        "entity_types": entity_types,
        "test_loader": test_loader,
        "best_dev_f1": float(best_dev_score),
        "best_epoch": int(best_epoch),
        "best_dev_metrics": best_dev_metrics or {},
        "test_seqeval": test_seqeval,
        "test_nervaluate": test_nervaluate,
    }

def run_optuna(
    architecture: str,
    train_examples: List[Dict],
    dev_examples: List[Dict],
    base_config: TrainConfig,
    device: torch.device,
    logger: logging.Logger,
    save_dir: Path,
    n_trials: int,
    fresh: bool = False,
    backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> "optuna.Study":
    if optuna is None:
        raise RuntimeError("optuna is required")

    arch_dir = save_dir / architecture
    arch_dir.mkdir(parents=True, exist_ok=True)

    db_path = arch_dir / "optuna.db"
    if fresh and db_path.exists():
        db_path.unlink()
        logger.info(f"[{architecture}] removed old optuna DB (--fresh_optuna)")

    def objective(trial):
        sampled = architecture_trial_space(trial, architecture)
        cfg = copy.deepcopy(base_config)
        cfg.architecture = architecture
        for k, v in sampled.items():
            setattr(cfg, k, v)
        result = train_one_run(
            train_examples, dev_examples, dev_examples,
            cfg, device, logger, trial,
            backbone_state_dict=backbone_state_dict,
        )
        return result["best_dev_f1"]

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=f"sqlite:///{db_path}",
        study_name=f"{architecture}_study",
        load_if_exists=not fresh,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    pd.DataFrame(study.trials_dataframe()).to_csv(arch_dir / "optuna_trials.csv", index=False)
    (arch_dir / "optuna_best.json").write_text(
        json.dumps(study.best_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"[{architecture}] optuna best params: {study.best_params}")
    return study

def extract_bert_backbone(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    backbone = {
        k[len("bert."):]: v.cpu()
        for k, v in state.items()
        if k.startswith("bert.")
    }
    if not backbone:
        raise RuntimeError(
            "Could not find 'bert.*' keys in model state dict. "
            "Make sure the model stores its encoder as self.bert."
        )
    return backbone

BASE_PARAMS = {
    "model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "input_scheme": DEFAULT_INPUT_SCHEME,
    "scheme": DEFAULT_OUTPUT_SCHEME,
    "max_length": 512,
    "batch_size": 4,
    "epochs": 15,
    "lr": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.15,
    "patience": 4,
    "grad_accum_steps": 1,
    "num_workers": 4,
    "use_amp": True,
    "focal_weight": 0.1,
    "focal_gamma": 2.0,
    "o_weight": 0.1,
    "label_smoothing": 0.0,
    "use_layer_pooling": True,
    "n_last_layers": 4,
    "mlp_hidden": 256,
    "gradient_checkpointing": True,
    "lstm_hidden": 256,
    "lstm_layers": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_classifier_type": "crf",
}

def parse_args():
    p = argparse.ArgumentParser(description="Cluster-aware (HDBSCAN) multi-architecture BiomedBERT NER training")

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
    p.add_argument("--architectures", nargs="+", default=DEFAULT_ARCHITECTURES,
                   choices=sorted(ARCHITECTURES.keys()))
    p.add_argument("--optuna_trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    p.add_argument("--selection_metric", default="strict_f1")
    p.add_argument("--fresh_optuna", action="store_true", default=False,
                   help="Delete existing Optuna DBs before search (clean start)")

    for k, v in BASE_PARAMS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(f"--{k}", type=float, default=v)
        else:
            p.add_argument(f"--{k}", default=v)

    return p.parse_args()

def _run_architecture(
    architecture: str,
    arch_idx: int,
    total_archs: int,
    args,
    base_params: Dict,
    train_ex: List[Dict],
    dev_ex: List[Dict],
    test_ex: List[Dict],
    device: torch.device,
    logger: logging.Logger,
    save_dir: Path,
    backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    logger.info("\n" + "=" * 80)
    logger.info(f"ARCHITECTURE {arch_idx}/{total_archs}: {architecture}")
    if architecture == "bert_lora" and backbone_state_dict is not None:
        logger.info("  >> LoRA warm-start: backbone loaded from best non-LoRA model")
    logger.info("=" * 80)

    arch_dir = save_dir / architecture
    arch_dir.mkdir(parents=True, exist_ok=True)

    arch_params = copy.deepcopy(base_params)
    arch_params.update(ARCHITECTURES[architecture])
    arch_params["architecture"] = architecture

    if args.optuna_trials > 0:
        study = run_optuna(
            architecture, train_ex, dev_ex,
            TrainConfig(**arch_params),
            device, logger, save_dir,
            args.optuna_trials,
            fresh=args.fresh_optuna,
            backbone_state_dict=backbone_state_dict,
        )
        arch_params.update(study.best_params)

    final_cfg = TrainConfig(**arch_params)
    result = train_one_run(
        train_ex, dev_ex, test_ex, final_cfg, device, logger,
        backbone_state_dict=backbone_state_dict,
    )

    row = {
        "architecture": architecture,
        "best_dev_f1": result["best_dev_f1"],
        "best_epoch": result["best_epoch"],
        "dev_f1": result["best_dev_metrics"].get("f1", float("nan")),
        "test_seqeval_f1": result["test_seqeval"].get("f1", float("nan")),
    }
    for k, v in result["test_nervaluate"].items():
        row[f"test_{k}"] = v

    logger.info(f"[{architecture}] final row: {row}")

    ckpt = arch_dir / f"best_{architecture}.pt"
    torch.save(
        {
            "model_state_dict": result["model"].state_dict(),
            "label2id": result["label2id"],
            "id2label": result["id2label"],
            "entity_types": result["entity_types"],
            "config": asdict(final_cfg),
            "metrics": row,
        },
        ckpt,
    )

    (arch_dir / f"best_{architecture}.meta.json").write_text(
        json.dumps(
            {"architecture": architecture, "metrics": row, "config": asdict(final_cfg)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame([row]).to_csv(arch_dir / "final_metrics.csv", index=False)
    return {"row": row, "model": result["model"]}

def main():
    args = parse_args()
    save_dir = Path(os.path.expanduser(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_ex, dev_ex, test_ex = load_and_split(
        data_path=Path(os.path.expanduser(args.data_path)),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        logger=logger,
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
    )

    base_params = {k: getattr(args, k) for k in BASE_PARAMS}

    requested = list(args.architectures)
    non_lora = [a for a in requested if a != "bert_lora"]
    lora = [a for a in requested if a == "bert_lora"]
    ordered_archs = non_lora + lora
    total = len(ordered_archs)

    all_rows: List[Dict] = []
    best_per_arch: Dict[str, Dict] = {}
    non_lora_models: Dict[str, nn.Module] = {}

    for arch_idx, architecture in enumerate(non_lora, 1):
        out = _run_architecture(
            architecture=architecture, arch_idx=arch_idx, total_archs=total,
            args=args, base_params=base_params,
            train_ex=train_ex, dev_ex=dev_ex, test_ex=test_ex,
            device=device, logger=logger, save_dir=save_dir,
            backbone_state_dict=None,
        )
        row = out["row"]
        all_rows.append(row)
        best_per_arch[architecture] = row
        non_lora_models[architecture] = out["model"]

    if lora:
        backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None

        if non_lora_models:
            LORA_PREFERRED_DONORS = ["bert_crf", "bert_linear", "bert_bilstm_crf"]
            donor_arch = next(
                (a for a in LORA_PREFERRED_DONORS if a in non_lora_models),
                max(non_lora_models.keys(), key=lambda a: best_per_arch[a]["best_dev_f1"]),
            )
            logger.info(
                f"\n[LoRA warm-start] Donor: {donor_arch} "
                f"(dev_f1={best_per_arch[donor_arch]['best_dev_f1']:.4f}, topology-compatible). "
                f"Extracting BERT backbone."
            )
            backbone_state_dict = extract_bert_backbone(non_lora_models[donor_arch])
            for m in non_lora_models.values():
                m.cpu()
            non_lora_models.clear()
        else:
            logger.info(
                "[LoRA warm-start] No non-LoRA architectures in this run; "
                "starting from pre-trained HuggingFace checkpoint."
            )

        for arch_idx, architecture in enumerate(lora, len(non_lora) + 1):
            out = _run_architecture(
                architecture=architecture, arch_idx=arch_idx, total_archs=total,
                args=args, base_params=base_params,
                train_ex=train_ex, dev_ex=dev_ex, test_ex=test_ex,
                device=device, logger=logger, save_dir=save_dir,
                backbone_state_dict=backbone_state_dict,
            )
            row = out["row"]
            all_rows.append(row)
            best_per_arch[architecture] = row

    pd.DataFrame(all_rows).to_csv(save_dir / "all_runs.csv", index=False)
    (save_dir / "best_by_architecture.json").write_text(
        json.dumps(best_per_arch, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)

    metric_col_map = {
        "strict_f1":   "test_strict_f1",
        "partial_f1":  "test_partial_f1",
        "exact_f1":    "test_exact_f1",
        "ent_type_f1": "test_ent_type_f1",
        "f1":          "test_seqeval_f1",
    }
    sel_col = metric_col_map.get(args.selection_metric, f"test_{args.selection_metric}")

    for arch, info in best_per_arch.items():
        sel_val = info.get(sel_col, float("nan"))
        logger.info(
            f"{arch}: best_dev_f1={info['best_dev_f1']:.4f} | "
            f"test_seqeval_f1={info['test_seqeval_f1']:.4f} | "
            f"{sel_col}={sel_val:.4f}"
        )

    valid = {a: i for a, i in best_per_arch.items()
             if not math.isnan(i.get(sel_col, float("nan")))}
    if valid:
        overall_best = max(valid, key=lambda a: valid[a][sel_col])
        logger.info(
            f"\nBest architecture by '{sel_col}': {overall_best} "
            f"({sel_col}={best_per_arch[overall_best][sel_col]:.4f})"
        )
        (save_dir / "best_overall.json").write_text(
            json.dumps(
                {"architecture": overall_best, "selection_metric": sel_col,
                 "metrics": best_per_arch[overall_best]},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    logger.info(f"Done: {save_dir}")

if __name__ == "__main__":
    main()
