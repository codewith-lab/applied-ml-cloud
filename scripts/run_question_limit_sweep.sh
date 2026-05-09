#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_question_limit_sweep.sh graph_hybrid
#   scripts/run_question_limit_sweep.sh graph_hybrid_lora
#   scripts/run_question_limit_sweep.sh dense_bm25_ce
#   scripts/run_question_limit_sweep.sh dense_bm25_lora
#   scripts/run_question_limit_sweep.sh dense_graph
#
# Optional env vars:
#   QUESTION_LIMITS="50 100 150"
#   DATASET_NAME="PatronusAI/financebench"
#   DATASET_SPLIT="train"
#   DB="artifacts/graph.sqlite"
#   INDEX_DIR="artifacts/index"
#   RESULTS="artifacts/results"
#   USE_EVAL_CACHE=1            # set to 0 to force recomputation
#   EVAL_CACHE_DIR="artifacts/results/cache"
#   DOC_SCOPE="strict"          # strict|fallback|off; strict keeps FinanceBench retrieval inside the evidence document

VARIANT=${1:-graph_hybrid}
QUESTION_LIMITS=${QUESTION_LIMITS:-"50 100 150"}
DATASET_NAME=${DATASET_NAME:-PatronusAI/financebench}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DB=${DB:-artifacts/graph.sqlite}
INDEX_DIR=${INDEX_DIR:-artifacts/index}
RESULTS=${RESULTS:-artifacts/results}
USE_EVAL_CACHE=${USE_EVAL_CACHE:-1}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-$RESULTS/cache}
DOC_SCOPE=${DOC_SCOPE:-strict}
TOP_K=${TOP_K:-8}
RERANK_TOP_N=${RERANK_TOP_N:-12}

CACHE_ARGS=()
if [[ "$USE_EVAL_CACHE" == "1" ]]; then
  CACHE_ARGS+=(--use-eval-cache --eval-cache-dir "$EVAL_CACHE_DIR")
else
  CACHE_ARGS+=(--no-use-eval-cache)
fi

for N in $QUESTION_LIMITS; do
  OUT_DIR="$RESULTS/e2e_${VARIANT}_${N}q"
  echo "Running $VARIANT on first $N FinanceBench questions -> $OUT_DIR"
  fin-graph-rag eval-financebench \
    --dataset-name "$DATASET_NAME" \
    --split "$DATASET_SPLIT" \
    --max-questions "$N" \
    --variant "$VARIANT" \
    --doc-scope "$DOC_SCOPE" \
    --top-k "$TOP_K" \
    --rerank-top-n "$RERANK_TOP_N" \
    --db "$DB" \
    --index-dir "$INDEX_DIR" \
    --out-dir "$OUT_DIR" \
    "${CACHE_ARGS[@]}"
done
