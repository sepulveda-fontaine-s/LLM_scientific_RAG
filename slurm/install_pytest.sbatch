#!/bin/bash

#SBATCH --job-name=install_pytest
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=install_pytest_%j.out
#SBATCH --error=install_pytest_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

pip install -e .

python -c "import pytest; print('pytest:', pytest.__version__)"

pip check