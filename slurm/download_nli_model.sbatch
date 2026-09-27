#!/bin/bash

#SBATCH --job-name=rag_nli_download
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=nli_download_%j.out
#SBATCH --error=nli_download_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cross-encoder/nli-deberta-v3-small', local_dir='models/validators/nli-deberta-v3-small')"