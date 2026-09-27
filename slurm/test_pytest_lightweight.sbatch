#!/bin/bash

#SBATCH --job-name=pytest_light
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=pytest_light_%j.out
#SBATCH --error=pytest_light_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

pytest -q -m "not integration" tests