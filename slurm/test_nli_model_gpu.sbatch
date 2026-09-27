#!/bin/bash

#SBATCH --job-name=rag_nli_test
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=nli_test_%j.out
#SBATCH --error=nli_test_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/test_nli_model.py