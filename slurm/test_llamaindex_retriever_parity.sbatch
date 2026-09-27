#!/bin/bash

#SBATCH --job-name=li_parity
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=li_parity_%j.out
#SBATCH --error=li_parity_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

python scripts/test_llamaindex_retriever_parity.py