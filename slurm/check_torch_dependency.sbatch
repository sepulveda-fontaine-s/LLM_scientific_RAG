#!/bin/bash

#SBATCH --job-name=rag_check_torch_dep
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_torch_dep_%j.out
#SBATCH --error=check_torch_dep_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "===== sentence-transformers ====="
python -m pip show sentence-transformers

echo
echo "===== torch ====="
python -m pip show torch