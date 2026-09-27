#!/bin/bash

#SBATCH --job-name=rag_qwen_test
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=qwen_test_%j.out
#SBATCH --error=qwen_test_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

echo "===== GPU BEFORE MODEL LOAD ====="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv

echo
echo "===== QWEN GENERATION TEST ====="

python scripts/test_generator.py

echo
echo "===== GPU AFTER TEST ====="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv