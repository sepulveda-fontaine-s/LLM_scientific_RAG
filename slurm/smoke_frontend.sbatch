#!/bin/bash

#SBATCH --job-name=smoke_frontend
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=smoke_frontend_%j.out
#SBATCH --error=smoke_frontend_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

cleanup() {
    kill "$API_PID" "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup EXIT

uvicorn llm_rag.api:app \
    --host 127.0.0.1 \
    --port 8765 \
    > "frontend_api_${SLURM_JOB_ID}.log" 2>&1 &

API_PID=$!

streamlit run frontend/app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true \
    > "frontend_streamlit_${SLURM_JOB_ID}.log" 2>&1 &

FRONTEND_PID=$!

sleep 20

python - <<'PY'
import urllib.request

for name, url in [
    ("FastAPI", "http://127.0.0.1:8765/health"),
    ("Streamlit", "http://127.0.0.1:8501"),
]:
    with urllib.request.urlopen(url, timeout=10) as response:
        print(f"{name}: HTTP {response.status}")

print("FRONTEND SMOKE TEST: PASS")
PY