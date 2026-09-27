#!/bin/bash

#SBATCH --job-name=rag_check_torch_versions
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_torch_versions_%j.out
#SBATCH --error=check_torch_versions_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -m pip index versions torch \
    --index-url https://download.pytorch.org/whl/cu128