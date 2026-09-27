#!/bin/bash

#SBATCH --job-name=rag_cuda_72g
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=cuda_72g_%j.out
#SBATCH --error=cuda_72g_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "===== LIMITS ====="
ulimit -a

echo
echo "===== GPU BEFORE TORCH ====="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv

echo
echo "===== PYTORCH CUDA TEST ====="

python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

torch.cuda.init()

print("CUDA initialized: OK")
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY