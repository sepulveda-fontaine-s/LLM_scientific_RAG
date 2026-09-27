#!/bin/bash

#SBATCH --job-name=rag_create_venv
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=create_venv_%j.out
#SBATCH --error=create_venv_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

# Create the environment using Python 3.11 from the compute node.
# --copies avoids creating symlinks back to the node-specific /usr/bin/python3.
python3 -m venv --copies .venv

source .venv/bin/activate

python --version
which python