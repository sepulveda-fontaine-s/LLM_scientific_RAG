#!/bin/bash

#SBATCH --job-name=rag_eval_validate
#SBATCH --partition=CLUSTER
#SBATCH --qos=normal
#SBATCH --exclude=i5-01,i5-02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=eval_validate_%j.out
#SBATCH --error=eval_validate_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

module load Python3/3.10.5

python3 scripts/validate_retrieval_eval.py