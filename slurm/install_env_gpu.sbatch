#!/bin/bash

#SBATCH --job-name=rag_install_env
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=install_env_%j.out
#SBATCH --error=install_env_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check

echo
echo "Environment installation complete."
python --version