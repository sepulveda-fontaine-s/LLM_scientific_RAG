#!/bin/bash

#SBATCH --job-name=rag_check_cuda
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=check_cuda_%j.out
#SBATCH --error=check_cuda_%j.err

echo "===== NVIDIA DRIVER ====="
nvidia-smi

echo
echo "===== CUDA MODULES ====="
module avail cuda 2>&1

echo
echo "===== CUDA SPIDER ====="
module spider cuda 2>&1