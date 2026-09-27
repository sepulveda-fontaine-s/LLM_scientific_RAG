#!/bin/bash

#SBATCH --job-name=fastapi_smoke
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=fastapi_smoke_%j.out
#SBATCH --error=fastapi_smoke_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

python -m uvicorn \
    llm_rag.api:app \
    --host 127.0.0.1 \
    --port 8765 \
    > fastapi_server_${SLURM_JOB_ID}.log 2>&1 &

SERVER_PID=$!

trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

python scripts/smoke_fastapi.py