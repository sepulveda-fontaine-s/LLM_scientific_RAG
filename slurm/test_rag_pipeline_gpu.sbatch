#!/bin/bash

#SBATCH --job-name=rag_end_to_end
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=rag_end_to_end_%j.out
#SBATCH --error=rag_end_to_end_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,memory.free \
    --format=csv

echo
echo "===== END-TO-END RAG ====="

python scripts/test_rag_pipeline.py