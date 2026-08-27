# Biomedical NER Pipeline

A pipeline for downloading biomedical assays from ChEMBL, performing semantic clustering of assay descriptions, training NER models with cluster-aware data splitting, and analyzing corpus coverage.

## Repository Structure

| File                                   | Description                                                                                                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `download_chembl_assays.py`            | Downloads assays from the ChEMBL REST API (types F/B, `confidence_score=9`) into SQLite, with resume support and TSV export.                                       |
| `analyze_semantic_coverage.py`         | Analyzes semantic corpus coverage by comparing assays with the training corpus using sentence embeddings (BioSimCSE) and calibrated similarity thresholds.         |
| `cluster_uncovered_assays.py`          | Clusters uncovered assays using UMAP + HDBSCAN and extracts cluster medoids for manual NER annotation.                                                             |
| `analyze_hdbscan_params.py`            | Performs HDBSCAN parameter sensitivity analysis for `min_cluster_size` and `min_samples`, with visualization.                                                      |
| `visualize_hdbscan_embeddings.py`      | Generates 2D/3D visualizations of embeddings colored by HDBSCAN cluster assignments (static PNG and optional interactive HTML).                                    |
| `visualize_clusters_2d.py`             | Creates a compact 2D visualization of all clusters using distinct colors.                                                                                          |
| `visualize_clusters_3d.py`             | Generates a static 3D visualization of all clusters using Matplotlib.                                                                                              |
| `visualize_clusters_3d_interactive.py` | Generates an interactive 3D cluster visualization using Plotly.                                                                                                    |
| `train_ner_base.py`                    | Base NER training module with model definitions, tokenization, evaluation metrics, Optuna hyperparameter optimization, and MiniBatchKMeans-based data splitting.   |
| `train_ner_cluster_aware.py`           | Trains NER models based on BiomedBERT using cluster-aware splits derived from UMAP + HDBSCAN; supports Optuna, multiple architectures, and LoRA warm-start.        |
| `compare_split_strategies.py`          | Compares data-splitting strategies (`random`, `stratified`, `cluster_stratified`, `cluster_aware`) through data-leakage simulations and a `bert_vanilla` baseline. |

## NER Architectures

* `bert_linear` — BERT + Linear classifier
* `bert_crf` — BERT + CRF with layer pooling and MLP
* `bert_bilstm_crf` — BERT + BiLSTM + CRF
* `bert_lora` — LoRA applied to BERT with a CRF/Linear classifier
* `bert_vanilla` — Vanilla BERT + Linear classifier without class weighting, focal loss, or label smoothing; used as a baseline

## Pipeline Execution

### 1. Download Assay Data

```bash
python download_chembl_assays.py --db bronze_assays.db --tsv bronze_assays.tsv
```

### 2. Semantic Coverage Analysis

```bash
python analyze_semantic_coverage.py \
    --train-json train.json \
    --assays-db bronze_assays.db \
    --out-prefix coverage_report
```

### 3. Cluster Uncovered Assays for Annotation

```bash
python cluster_uncovered_assays.py \
    --train-json train.json \
    --assays-db bronze_assays.db \
    --assays-text-column description \
    --covered-thr 0.8429 \
    --out-prefix logs/annotation_candidates
```

### 4. Clustering and Visualization

```bash
python visualize_hdbscan_embeddings.py \
    --vecs-cache embeddings.npy \
    --out-dir figures

python visualize_clusters_2d.py \
    --vecs-cache embeddings.npy \
    --out-dir figures

python visualize_clusters_3d.py \
    --vecs-cache embeddings.npy \
    --out-dir figures
```

### 5. HDBSCAN Parameter Sensitivity Analysis

```bash
python analyze_hdbscan_params.py \
    --vecs-cache embeddings.npy \
    --out-dir figures \
    --min-cluster-sizes 10 15 20 25 30 40 50 70 100 \
    --min-samples-list 3 5 8 12
```

### 6. NER Training

```bash
python train_ner_cluster_aware.py \
    --data_path train.json \
    --save_dir experiments/run1 \
    --vecs-cache embeddings.npy \
    --architectures bert_linear bert_crf bert_bilstm_crf bert_lora
```

### 7. Compare Data-Splitting Strategies

```bash
python compare_split_strategies.py \
    --data_path train.json \
    --save_dir experiments/leakage \
    --vecs-cache embeddings.npy
```

## Dependencies

See [`requirements.txt`](requirements.txt) for the complete list of dependencies.
