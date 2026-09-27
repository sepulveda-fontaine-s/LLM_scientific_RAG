#!/bin/bash

#SBATCH --job-name=rag_gen_eval_validate
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=gen_eval_validate_%j.out
#SBATCH --error=gen_eval_validate_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

/usr/bin/python3 scripts/validate_generation_eval.py