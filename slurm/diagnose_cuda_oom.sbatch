#!/bin/bash

#SBATCH --job-name=rag_cuda_oom_diag
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=cuda_oom_diag_%j.out
#SBATCH --error=cuda_oom_diag_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "===== NODE ====="
hostname

echo
echo "===== SLURM GPU VARIABLES ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_GPUS=${SLURM_GPUS}"

echo
echo "===== NVIDIA-SMI ====="
nvidia-smi

echo
echo "===== GPU MEMORY ====="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv

echo
echo "===== HOST MEMORY ====="
free -h

echo
echo "===== LIMITS ====="
ulimit -a

echo
echo "===== CUDA LIBRARY ENV ====="
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

echo
echo "===== PYTORCH ====="
python - <<'PY'
import os
import torch

print("Torch:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

try:
    print("Device count:", torch.cuda.device_count())
except Exception as exc:
    print("device_count exception:", repr(exc))

try:
    torch.cuda.init()
    print("torch.cuda.init(): OK")
except Exception as exc:
    print("torch.cuda.init() exception:", repr(exc))
PY