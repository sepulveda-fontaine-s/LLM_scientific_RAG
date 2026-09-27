#!/bin/bash

#SBATCH --job-name=rag_torch_gpu
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mail-user=salome.sepulveda.fontaine@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=torch_gpu_%j.out
#SBATCH --error=torch_gpu_%j.err

set -e

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
