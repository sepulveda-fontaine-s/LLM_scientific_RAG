#!/bin/bash

#SBATCH --job-name=rag_check_python
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_python_%j.out
#SBATCH --error=check_python_%j.err

echo "===== SYSTEM PYTHON ====="
which python3 || true
python3 --version || true

echo
echo "===== PYTHON 3.12 ====="
which python3.12 || true
python3.12 --version || true

echo
echo "===== MODULES ====="
module avail python 2>&1 || true