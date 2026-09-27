#!/bin/bash

#SBATCH --job-name=pytest_citations
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=pytest_citations_%j.out
#SBATCH --error=pytest_citations_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

pytest -q tests/test_citation_validator.py