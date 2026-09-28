#!/bin/bash
#SBATCH --job-name=compare_split_strategies
#SBATCH --partition=6000-ada
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
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

python3 compare_split_strategies.py \
    --data_path ~/project/data/biomedbert_merged_v3_clean_4.json \
    --save_dir experiments/split_ablation \
    --vecs_cache ~/project/embeddings_a03ce33c.npy

echo "Готово. Смотри experiments/split_ablation/"