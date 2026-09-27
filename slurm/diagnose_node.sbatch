#!/bin/bash

#SBATCH --job-name=rag_diag
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=rag_diag_%j.out
#SBATCH --error=rag_diag_%j.err

# Stop immediately if any command fails.
set -e

echo "===== JOB INFORMATION ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"

echo
echo "===== CPU ====="
echo "SLURM CPUs assigned: ${SLURM_CPUS_PER_TASK}"
lscpu | grep -E "^CPU\(s\)|Model name"

echo
echo "===== MEMORY ====="
free -h

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== GPU SUMMARY ====="
nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv

echo
echo "===== CUDA ====="
if command -v nvcc > /dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc not available in PATH"
fi

echo
echo "===== PYTHON ====="
python3 --version || true
which python3 || true

echo
echo "===== INTERNET TEST: HUGGING FACE ====="

if command -v curl > /dev/null 2>&1; then
    curl -I --max-time 10 https://huggingface.co || true
else
    echo "curl is not installed; Internet test skipped."
fi

echo
echo "===== DIAGNOSTIC COMPLETE ====="