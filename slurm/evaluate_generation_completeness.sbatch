#!/bin/bash

#SBATCH --job-name=rag_gen_complete
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=gen_complete_%j.out
#SBATCH --error=gen_complete_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python scripts/evaluate_generation_completeness.py