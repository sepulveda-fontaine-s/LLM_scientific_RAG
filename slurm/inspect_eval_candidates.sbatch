#!/bin/bash

#SBATCH --job-name=rag_holdout_candidates
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=holdout_candidates_%j.out
#SBATCH --error=holdout_candidates_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

echo "===== SENTENCE-BERT PAGES 5-7 ====="

/usr/bin/python3 scripts/inspect_eval_candidates.py \
    --source "Sentence_BERT" \
    --start-page 5 \
    --end-page 7

echo ""
echo "===== RAG SURVEY PAGES 6-8 ====="

/usr/bin/python3 scripts/inspect_eval_candidates.py \
    --source "RAG_for_LLMs" \
    --start-page 6 \
    --end-page 8