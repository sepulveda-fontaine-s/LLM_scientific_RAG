#!/bin/bash

#SBATCH --job-name=rag_debug_eval_ids
#SBATCH --partition=CLUSTER
#SBATCH --qos=normal
#SBATCH --exclude=i5-01,i5-02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=debug_eval_ids_%j.out
#SBATCH --error=debug_eval_ids_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

module load Python3/3.10.5

python3 scripts/debug_eval_ids.py