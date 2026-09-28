#!/bin/bash
#SBATCH --job-name=compare_split_hdbscan
#SBATCH --partition=6000-ada
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/home/%u/work/biomedical-ner-pipeline/logs/%x-%j.out
#SBATCH --error=/home/%u/work/biomedical-ner-pipeline/logs/%x-%j.err

set -eo pipefail
mkdir -p ~/work/biomedical-ner-pipeline/logs
cd ~/work/biomedical-ner-pipeline

set +u
source ~/project/venv/bin/activate
set -u
unset LD_LIBRARY_PATH

echo "Using python: $(which python3)"
python3 -c "import torch, transformers, umap, sklearn, peft, optuna, nervaluate; from torchcrf import CRF; print('cuda:', torch.cuda.is_available()); print('deps OK')"

# ВАЖНО: --optuna_trials 0 по умолчанию — иначе это 4 стратегии x 5 архитектур x
# N trials, что в разы дороже, чем всё, что мы гоняли раньше. Начинаем с
# одного honest-прогона на BASE_PARAMS по умолчанию (без подбора гиперпараметров),
# чтобы быстро увидеть картину по всем 4 стратегиям и 5 архитектурам,
# и только потом решаем, нужен ли Optuna поверх этого.
python3 compare_split_strategies.py \
    --data_path ~/project/data/biomedbert_merged_v3_clean_4.json \
    --save_dir experiments/leakage_hdbscan \
    --vecs_cache ~/project/embeddings_a03ce33c.npy \
    --num_workers 0 \
    --optuna_trials 0 \
    --architectures bert_vanilla bert_linear bert_crf bert_bilstm_crf bert_lora \
    --strategies 01_random 02_stratified_entity 03_cluster_stratified 04_cluster_aware \
    --model_name microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract \
    --max_length 512 \
    --batch_size 2 \
    --grad_accum_steps 8

echo "Готово. Смотри experiments/leakage_hdbscan/"
