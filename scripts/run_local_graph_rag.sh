#!/usr/bin/env bash
set -euo pipefail

PDF_DIR=${PDF_DIR:-data/pdfs}
DB=${DB:-artifacts/graph.sqlite}
INDEX_DIR=${INDEX_DIR:-artifacts/index}
RESULTS=${RESULTS:-artifacts/results}
MAX_QUESTIONS=${MAX_QUESTIONS:-150}

fin-graph-rag ingest --pdf-dir "$PDF_DIR" --db "$DB"
fin-graph-rag build-indexes --db "$DB" --index-dir "$INDEX_DIR"

fin-graph-rag retrieve "What risks could pressure gross margin?" \
  --db "$DB" \
  --index-dir "$INDEX_DIR" \
  --variant graph_hybrid \
  --top-k 12

fin-graph-rag eval-financebench \
  --variant graph_hybrid \
  --max-questions "$MAX_QUESTIONS" \
  --db "$DB" \
  --index-dir "$INDEX_DIR" \
  --out-dir "$RESULTS/e2e_graph_hybrid_${MAX_QUESTIONS}q"

fin-graph-rag judge-personas \
  --db "$DB" \
  --index-dir "$INDEX_DIR" \
  --num-personas 2 \
  --questions-per-task 2 \
  --out-dir "$RESULTS/persona_judge"
