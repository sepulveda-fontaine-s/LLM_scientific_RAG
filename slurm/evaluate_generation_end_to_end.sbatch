#!/bin/bash

#SBATCH --job-name=rag_gen_e2e
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=gen_e2e_%j.out
#SBATCH --error=gen_e2e_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

python scripts/evaluate_generation_end_to_end.py