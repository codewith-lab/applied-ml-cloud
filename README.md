# FinRAG

Local GraphRAG pipeline for SEC-style financial filing QA, with companion LoRA reranking and GKE deployment assets. The project uses:

- **SQLite** as the fast local retrieval graph store.
- **BM25 + FAISS dense retrieval** over the same parsed `Block.text` corpus.
- **Neo4j mirror** for graph visualization.
- **CE / LoRA reranking** for the accuracy-vs-latency comparison.
- **Document-scoped FinanceBench evaluation**, using each row's `doc_link`, `company`, `doc_period`, and `doc_type` to restrict retrieval to the correct filing.
- **Standalone LoRA / ranking experiments** under `src/fin_graph_rag/lora`.
- **GKE cloud deployment experimentss** under `src/fin_graph_rag/cloud`.

---

## 0. Project layout

```text
applied-ml-cloud/
  README.md
  pyproject.toml
  configs/default.yaml
  data/pdfs/                         # FinanceBench PDFs go here
  adaptor/                           # local LoRA adapter; adapter/ also works
  artifacts/
    graph.sqlite                     # built by ingest
    index/                           # BM25 + FAISS indexes
    results/                         # eval outputs, cache, plots
  scripts/
    run_question_limit_sweep.sh
    run_slide_model_sweep.sh
    run_local_graph_rag.sh
  src/fin_graph_rag/
    ingest/                          # PDF parse, SQLite graph, Neo4j sync
    indexing/                        # BM25 + dense indexes
    retrieval/                       # dense/BM25/graph retrieval and merge
    rerank/                          # CE and runtime LoRA rerankers
    evaluation/                      # FinanceBench, plots, LLM-as-judge
    prompts/claim_extraction.txt     # only used by --use-llm-claims
    lora/                            # Open WebUI demo + standalone reranker work
      demo/                          # FinRAG Open WebUI function and screenshots
      reranker/                      # LoRA/ranker scripts, data, cached outputs
    cloud/                           # GKE, vLLM, Neo4j, Streamlit
```

The top-level CLI is the local retrieval/evaluation path. The `lora/` and `cloud/` folders are companion assets for the demo, standalone fine-tuning/ranking experiments, and cloud deployment story.

---

## 1. Install

Use a clean virtual environment. On macOS, avoid mixing Conda `base` with pip-installed Torch if possible.

```bash
# From the project root.
cd applied-ml-cloud

# Optional but recommended if your prompt shows both (.venv) and (base).
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[lora,neo4j]'

# Confirm the CLI points to this project, not Anaconda.
which python
which fin-graph-rag
```

Expected CLI path:

```text
.../applied-ml-cloud/.venv/bin/fin-graph-rag
```

If the CLI points to `/opt/anaconda3/bin/fin-graph-rag`, reinstall inside the active venv and clear the shell cache:

```bash
python -m pip install -e '.[lora,neo4j]'
hash -r
```

Standalone scripts in `src/fin_graph_rag/lora/reranker` have their own heavier experiment dependencies:

```bash
python -m pip install -r src/fin_graph_rag/lora/reranker/requirements-lora.txt
```

---

## 2. Prepare data and model files

### 2.1 Put PDFs in `data/pdfs/`

```bash
mkdir -p data/pdfs
cp /path/to/financebench/pdfs/*.pdf data/pdfs/
```

FinanceBench strict document scope works best when filenames contain recognizable metadata, for example:

```text
3M_2018_10K.pdf
MMM_2018_10K.pdf
0001558370-19-000470.pdf
```

During eval, the matcher uses FinanceBench row fields such as:

```json
{
  "company": "3M",
  "doc_type": "10k",
  "doc_period": 2018,
  "doc_link": "https://.../0001558370-19-000470.pdf"
}
```

It creates hints from `company`, `ticker`, `doc_period`, `doc_type`, and the URL basename in `doc_link`, then restricts retrieval to the matched filing.

### 2.2 Put the LoRA adapter in `./adaptor`

The runtime retrieval pipeline assumes your fine-tuned reranker adapter is local:

```text
applied-ml-cloud/adaptor/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  tokenizer_config.json
```

Set:

```bash
export LORA_BASE_MODEL="unsloth/Llama-3.2-1B-Instruct"
export LORA_ADAPTER_PATH="./adaptor"  # use ./adapter only if your folder is named adapter
```

Validate the adapter weights:

```bash
python - <<'PY'
from safetensors import safe_open
path = "./adaptor/adapter_model.safetensors"
with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print("OK:", path)
print("num tensors:", len(keys))
print("score keys:", [k for k in keys if "score" in k.lower() or "classifier" in k.lower()])
print("first keys:", keys[:10])
PY
```

A valid sequence-classification reranker adapter should include a score/classifier head, for example:

```text
base_model.model.score.weight
```

---

## 3. Build the local retrieval graph

This parses PDFs into `Document -> Page -> Block`, creates table nodes, extracts cheap regex risk/P&L claims, and writes everything to SQLite.

```bash
fin-graph-rag ingest \
  --pdf-dir data/pdfs \
  --db artifacts/graph.sqlite
```

The graph contains:

```text
(:Document)-[:HAS_PAGE]->(:Page)-[:HAS_BLOCK]->(:Block)
(:Block)-[:NEXT_BLOCK|PREV_BLOCK]->(:Block)
(:Block)-[:IN_SECTION]->(:Section)
(:Section)-[:HAS_SECTION_BLOCK]->(:Block)
(:Block)-[:MENTIONS_TABLE]->(:Table)
(:Page)-[:HAS_TABLE]->(:Table)
(:Block)-[:SUPPORTS_CLAIM]->(:CausalClaim)
(:CausalClaim)-[:MENTIONS_RISK]->(:RiskEntity)
(:CausalClaim)-[:IMPACTS_PNL]->(:PnlDriver)
```

`Block` is the retrieval unit. `Table` is a metadata/visualization node linked from table-like blocks.

### Optional LLM claim extraction

```bash
export OPENAI_API_KEY="YOUR_KEY"
export ANSWER_MODEL="gpt-4o-mini"

fin-graph-rag ingest \
  --pdf-dir data/pdfs \
  --db artifacts/graph.sqlite \
  --use-llm-claims \
  --max-claim-blocks 200
```

Prompt file:

```text
src/fin_graph_rag/prompts/claim_extraction.txt
```

The LLM is asked to return strict JSON claims:

```json
{
  "claims": [
    {
      "claim_text": "...",
      "risk_entity": "foreign currency",
      "pnl_driver": "gross margin",
      "direction": "decreases",
      "confidence": 0.8
    }
  ]
}
```

---

## 4. Build BM25 and dense indexes

It indexes the same `Block.text` rows from SQLite.

```bash
fin-graph-rag build-indexes \
  --db artifacts/graph.sqlite \
  --index-dir artifacts/index
```

---

## 5. Mirror SQLite graph to Neo4j for visualization

Neo4j is a visualization mirror only. Evaluation and retrieval continue to use SQLite + BM25 + FAISS.

### 5.1 Start Neo4j on default ports

```bash
docker run --rm --name fin-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

Open:

```text
http://localhost:7474
```

Connect with:

```text
Connection URL: neo4j://localhost:7687
user: neo4j
password: password
```

### 5.2 Sync an existing SQLite graph to Neo4j

Use default port:

```bash
fin-graph-rag sync-neo4j \
  --db artifacts/graph.sqlite \
  --neo4j-uri bolt://127.0.0.1:7687 \
  --neo4j-user neo4j \
  --neo4j-password password
```

Use alternate port:

```bash
fin-graph-rag sync-neo4j \
  --db artifacts/graph.sqlite \
  --neo4j-uri bolt://127.0.0.1:7688 \
  --neo4j-user neo4j \
  --neo4j-password password
```

Useful Neo4j checks:

```cypher
MATCH (n)
RETURN labels(n) AS labels, count(*) AS n
ORDER BY n DESC;
```

```cypher
MATCH p=(d:Document)-[:HAS_PAGE]->(:Page)-[:HAS_BLOCK]->(:Block)-[:MENTIONS_TABLE]->(:Table)
RETURN p
LIMIT 50;
```

```cypher
MATCH p=(b:Block)-[:SUPPORTS_CLAIM]->(c:CausalClaim)-[:MENTIONS_RISK|IMPACTS_PNL]->(x)
RETURN p
LIMIT 50;
```

---

## 6. Retrieval tests

### 6.1 Non-LoRA GraphRAG retrieval

Use strict document scope when debugging a specific company/filing.

```bash
fin-graph-rag retrieve \
  "What risks could pressure gross margin for 3M?" \
  --db artifacts/graph.sqlite \
  --index-dir artifacts/index \
  --variant dense_graph \
  --doc-hint 3M \
  --doc-scope strict \
  --top-k 8
```

### 6.2 LoRA reranked retrieval

Use `--rerank-top-n` for the neural reranker pool and `--top-k` for final output size.

```bash
fin-graph-rag retrieve \
  "What risks could pressure gross margin for 3M?" \
  --db artifacts/graph.sqlite \
  --index-dir artifacts/index \
  --variant graph_hybrid_lora \
  --doc-hint 3M \
  --doc-scope strict \
  --lora-base-model "$LORA_BASE_MODEL" \
  --lora-adapter-path "$LORA_ADAPTER_PATH" \
  --lora-device cpu \
  --lora-dtype float32 \
  --lora-batch-size 1 \
  --lora-max-length 256 \
  --rerank-top-n 12 \
  --top-k 8
```

Meaning:

```text
--rerank-top-n 12   # send only 12 candidates to LoRA
--top-k 8           # return final 8 evidence blocks
```

---

## 7. Evaluation

These are the five variants used for the slide-style accuracy/latency plot.

| Slide label | CLI variant | Description |
|---|---|---|
| Dense | `dense` | FAISS dense retrieval only |
| Dense + BM25 | `dense_bm25` | Dense + BM25 reciprocal-rank fusion |
| Dense + BM25 + CE rerank | `dense_bm25_ce` | Generic cross-encoder reranking |
| Dense + BM25 + fine-tuned rerank | `dense_bm25_lora` | LoRA sequence-classification reranking |
| Dense + Graph | `dense_graph` | Dense seeds plus structural/semantic graph expansion |

```bash
export CE_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
export CE_DEVICE="cpu"

export LORA_BASE_MODEL="unsloth/Llama-3.2-1B-Instruct"
export LORA_ADAPTER_PATH="./adaptor"
```

### 7.1 Run 50-question evaluation

```bash
for v in dense dense_bm25 dense_bm25_ce dense_bm25_lora dense_graph; do
  fin-graph-rag eval-financebench \
    --variant "$v" \
    --max-questions 50 \
    --doc-scope strict \
    --db artifacts/graph.sqlite \
    --index-dir artifacts/index \
    --top-k 8 \
    --rerank-top-n 12 \
    --lora-base-model "$LORA_BASE_MODEL" \
    --lora-adapter-path "$LORA_ADAPTER_PATH" \
    --lora-device cpu \
    --lora-dtype float32 \
    --lora-batch-size 1 \
    --lora-max-length 256 \
    --out-dir "artifacts/results/e2e_${v}_50q" \
    --use-eval-cache \
    --eval-cache-dir artifacts/results/cache
done
```

### 7.2 Run full 150-question evaluation

The cache reuses rows already computed for 50/100-question runs.

```bash
for v in dense dense_bm25 dense_bm25_ce dense_bm25_lora dense_graph; do
  fin-graph-rag eval-financebench \
    --variant "$v" \
    --max-questions 150 \
    --doc-scope strict \
    --db artifacts/graph.sqlite \
    --index-dir artifacts/index \
    --top-k 8 \
    --rerank-top-n 12 \
    --lora-base-model "$LORA_BASE_MODEL" \
    --lora-adapter-path "$LORA_ADAPTER_PATH" \
    --lora-device cpu \
    --lora-dtype float32 \
    --lora-batch-size 1 \
    --lora-max-length 256 \
    --out-dir "artifacts/results/e2e_${v}_150q" \
    --use-eval-cache \
    --eval-cache-dir artifacts/results/cache
done
```

### 7.3 Plot accuracy vs latency

150-question plot:

```bash
fin-graph-rag plot-latency \
  --result-dirs artifacts/results/e2e_dense_150q \
  --result-dirs artifacts/results/e2e_dense_bm25_150q \
  --result-dirs artifacts/results/e2e_dense_bm25_ce_150q \
  --result-dirs artifacts/results/e2e_dense_bm25_lora_150q \
  --result-dirs artifacts/results/e2e_dense_graph_150q \
  --out artifacts/results/accuracy_latency_150q.png
```

`--result-dirs` must be repeated once per directory.

### 7.4 Eval cache

Cache location:

```text
artifacts/results/cache/
```

Run output location:

```text
artifacts/results/e2e_<variant>_<N>q/
```

With `--use-eval-cache`, increasing `--max-questions` from 50 to 100 should reuse the first 50 rows and compute only rows 51-100, as long as the cache key settings are unchanged.

Check cache files:

```bash
find artifacts/results/cache -type f | wc -l
find artifacts/results/cache -type f | sort | head
```

Force fresh computation:

```bash
rm -rf artifacts/results/cache
```

Delete the cache when you change retrieval settings, prompts, document scope, CE model, LoRA adapter, answer model, indexes, or PDFs.

---

## 8. LLM-as-judge risk/P&L causality evaluation

Set a real LLM endpoint. Without `OPENAI_API_KEY`, the project falls back to heuristics and should not be used for final judge metrics.

```bash
export OPENAI_API_KEY="YOUR_KEY"
export ANSWER_MODEL="gpt-4o-mini"
export JUDGE_MODEL="gpt-4o-mini"
```

Run the 2 persona x 2 task x 2 question judge setup:

```bash
fin-graph-rag judge-personas \
  --db artifacts/graph.sqlite \
  --index-dir artifacts/index \
  --num-personas 2 \
  --questions-per-task 2 \
  --out-dir artifacts/results/persona_judge_2x2
```

Outputs:

```text
artifacts/results/persona_judge_2x2/personas.json
artifacts/results/persona_judge_2x2/questions.jsonl
artifacts/results/persona_judge_2x2/comparisons.csv          # 8 raw comparisons
artifacts/results/persona_judge_2x2/preference_matrix.csv    # grouped 4-row summary
artifacts/results/persona_judge_2x2/summary.json
```

---

## 9. LoRA demo and standalone reranking assets

The `src/fin_graph_rag/lora` folder adds material that is separate from the main `fin-graph-rag` CLI:

```text
src/fin_graph_rag/lora/
  README.md                          # Open WebUI FinRAG demo walkthrough
  README_reranker.md                 # standalone reranker notes
  demo/
    finrag_demo.py                   # Open WebUI Function / Pipe named finrag
    granite_autogen_rag.py           # Granite AG2 baseline/reference
    image_researcher_granite_crewai.py
    lorademo*.png                    # demo screenshots
  reranker/
    lora_reranker.py                 # Llama sequence-classification LoRA trainer
    chunk_ranker.py                  # supervised chunk ranker
    chunk_best_ensemble.py           # cached ensemble materializer
    requirements-lora.txt
    outputs/                         # cached metrics/rankings/models
```

### 9.1 Open WebUI demo

The demo function is in:

```text
src/fin_graph_rag/lora/demo/finrag_demo.py
```

It exposes the selectable model ID `finrag` in Open WebUI and uses:

- built-in project context for explaining the FinRAG method, metrics, deployment, and future work;
- Open WebUI Knowledge search for user-uploaded SEC filings or graph-export chunks;
- Open WebUI web search for external references.

See the full demo setup in:

```text
src/fin_graph_rag/lora/README.md
```

### 9.2 Standalone LoRA reranker training

Install the standalone experiment dependencies:

```bash
cd src/fin_graph_rag/lora/reranker
python -m pip install -r requirements-lora.txt
```

Quick smoke run:

```bash
python lora_reranker.py \
  --epochs 1 \
  --doc-query-limit 30 \
  --chunk-query-limit -1 \
  --doc-val-queries 10 \
  --chunk-val-queries 0 \
  --max-train-pairs 120 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --output-dir outputs/llama3_smoke
```

The default base model is:

```text
unsloth/Llama-3.2-1B-Instruct
```
---

## 10. Cloud and GKE assets

The `src/fin_graph_rag/cloud` folder documents and stages the cloud version of the project: Neo4j, vLLM, a Streamlit/orchestrator frontend, GCS-backed artifacts, and GKE Jobs for ingestion and fine-tuning.

```text
src/fin_graph_rag/cloud/
  README.md                          # GKE GraphRAG deployment walkthrough
  cluster-provisioning/              # screenshots for cluster/GPU node setup
  infra-stateful-services/           # screenshots for Cloud SQL, GSA/KSA, Neo4j
  knowledge-graph-subsys/
    injest-job.yaml                  # ingestion Job manifest
  vllm-serving-subsys/
    vllm-inference-server.yaml       # vLLM OpenAI-compatible server
    orchestrator-deployment.yaml     # app/API deployment
    orchestrator-service.yaml        # ingress/service exposure
    Dockerfile.txt
  finetuning-subsys/
    data/                            # local copy of LoRA/ranking experiment data
    cloud/
      Dockerfile
      cloudbuild.yaml
      k8s/
        serviceaccount.yaml
        train-job-l4.yaml
        train-job-l4-ondemand.yaml
        train-job-t4.yaml
        merge-job.yaml
        merge-job-colab.yaml
        ensemble-job.yaml
      merge_lora_offline.py
    docs/cloud/                      # architecture images and evidence logs
  demo/                              # cloud demo screenshots
```

### 10.1 GKE GraphRAG serving path

Start with:

```text
src/fin_graph_rag/cloud/README.md
```

That walkthrough covers:

- Neo4j as the cloud knowledge graph;
- vLLM as an OpenAI-compatible inference server;
- Streamlit as the user-facing RAG assistant;
- GCS as the artifact/data bucket;
- Kubernetes ConfigMaps and Jobs for lightweight ingestion.

Important: replace placeholder project IDs, bucket names, image names, and passwords before running cloud manifests. The checked-in YAML and README values are class/demo defaults, not production secrets.

### 10.2 GKE fine-tuning path

The fine-tuning experiment covers:

- creating project variables and GCS buckets;
- enabling GCP APIs;
- building the trainer image with Cloud Build;
- creating a GKE Autopilot cluster;
- configuring Workload Identity;
- running L4 or T4 LoRA training Jobs;
- running a CPU merge Job to produce a merged model artifact;
- tearing the cluster down to stop billing.

Primary manifests:

```bash
kubectl apply -f src/fin_graph_rag/cloud/finetuning-subsys/cloud/k8s/serviceaccount.yaml
kubectl apply -f src/fin_graph_rag/cloud/finetuning-subsys/cloud/k8s/train-job-l4.yaml
kubectl apply -f src/fin_graph_rag/cloud/finetuning-subsys/cloud/k8s/train-job-t4.yaml
kubectl apply -f src/fin_graph_rag/cloud/finetuning-subsys/cloud/k8s/merge-job.yaml
```

The cloud evidence folder includes the same ensemble headline metrics as the local cached run:

```text
src/fin_graph_rag/cloud/finetuning-subsys/docs/cloud/cloud_evidence/metrics_best_ensemble.cloud.json
```
---

## 11. Developer smoke checks

If the package is installed into the active environment:

```bash
fin-graph-rag --help
python -m pytest
```

If you only want to run tests from a source checkout without installing the package:

```bash
PYTHONPATH=src python -m pytest
```

The lightweight unit tests cover document scoping, retrieval merge behavior, and stable text utilities.
