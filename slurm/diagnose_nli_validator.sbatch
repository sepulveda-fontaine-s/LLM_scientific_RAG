#!/bin/bash

#SBATCH --job-name=rag_nli_diag
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=nli_diag_%j.out
#SBATCH --error=nli_diag_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python scripts/diagnose_nli_windows.py