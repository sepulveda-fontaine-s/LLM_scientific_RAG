#!/bin/bash

#SBATCH --job-name=li_imports
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=li_imports_%j.out
#SBATCH --error=li_imports_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python -c "from llama_index.core.base.response.schema import Response; from llama_index.core.query_engine import CustomQueryEngine; from llama_index.core.retrievers import BaseRetriever; print('LlamaIndex query-engine imports: OK')"