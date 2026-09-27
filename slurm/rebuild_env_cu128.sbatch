#!/bin/bash

#SBATCH --job-name=rag_env_cu128
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=rebuild_env_%j.out
#SBATCH --error=rebuild_env_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

echo "===== RECREATE VENV ====="
rm -rf .venv
python3 -m venv --copies .venv
source .venv/bin/activate

python -m pip install --upgrade pip

echo
echo "===== INSTALL PYTORCH CUDA 12.8 ====="
python -m pip install "torch==2.11.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

echo
echo "===== INSTALL PROJECT ====="
python -m pip install -e .

echo
echo "===== DEPENDENCY CHECK ====="
python -m pip check

echo
echo "===== CUDA TEST ====="
python -c "import torch; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"