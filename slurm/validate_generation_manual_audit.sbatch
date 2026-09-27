#!/bin/bash

#SBATCH --job-name=rag_manual_audit
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=manual_audit_%j.out
#SBATCH --error=manual_audit_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

/usr/bin/python3 scripts/validate_generation_manual_audit.py