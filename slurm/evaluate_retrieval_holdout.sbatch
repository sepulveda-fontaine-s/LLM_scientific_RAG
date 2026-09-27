#!/bin/bash

#SBATCH --job-name=rag_holdout_eval
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=holdout_eval_%j.out
#SBATCH --error=holdout_eval_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

python scripts/evaluate_retrieval_holdout.py