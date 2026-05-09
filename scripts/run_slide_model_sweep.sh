#!/usr/bin/env bash
set -euo pipefail

# Runs the five slide-compatible variants:
#   Dense
#   Dense + BM25
#   Dense + BM25 + CE rerank
#   Dense + BM25 + fine-tuned rerank (LoRA)
#   Dense + Graph
#
# Usage:
#   QUESTION_LIMITS="25 50 100" scripts/run_slide_model_sweep.sh
#
# Required for LoRA variant:
#   export LORA_BASE_MODEL="unsloth/Llama-3.2-1B-Instruct"
#   export LORA_ADAPTER_PATH="./adaptor"   # or ./adapter
#
# Optional:
#   export CE_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
#   export CE_DEVICE=cpu
#   export DOC_SCOPE=strict
#   export USE_EVAL_CACHE=1

MODEL_VARIANTS=${MODEL_VARIANTS:-"dense dense_bm25 dense_bm25_ce dense_bm25_lora dense_graph"}
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

for VARIANT in $MODEL_VARIANTS; do
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
done
