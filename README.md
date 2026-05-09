# Fin GraphRAG for FinanceBench

Local GraphRAG pipeline for SEC-style financial filing QA. The project uses:

- **SQLite** as the fast local retrieval graph store.
- **BM25 + FAISS dense retrieval** over the same parsed `Block.text` corpus.
- **Optional Neo4j mirror** only for graph visualization.
- **Optional CE / LoRA reranking** for the accuracy-vs-latency comparison.
- **Document-scoped FinanceBench evaluation**, using each row's `doc_link`, `company`, `doc_period`, and `doc_type` to restrict retrieval to the correct filing.

Neo4j is not required for evaluation. Use it only when you want to inspect the graph visually.

---

## 0. Project layout

```text
fin_graph_rag_project/
  README.md
  pyproject.toml
  configs/default.yaml
  data/pdfs/                         # FinanceBench PDFs go here
  adaptor/                           # your LoRA adapter; adapter/ also works
  artifacts/
    graph.sqlite                     # built by ingest
    index/                           # BM25 + FAISS indexes
    results/                         # eval outputs, cache, plots
  scripts/
    run_question_limit_sweep.sh
    run_slide_model_sweep.sh
  src/fin_graph_rag/
    ingest/                          # PDF parse, SQLite graph, Neo4j sync
    indexing/                        # BM25 + dense indexes
    retrieval/                       # dense/BM25/graph retrieval and merge
    rerank/                          # CE and LoRA rerankers
    evaluation/                      # FinanceBench, plots, LLM-as-judge
    prompts/claim_extraction.txt     # only used by --use-llm-claims
```

---

## 1. Install

Use a clean virtual environment. On macOS, avoid mixing Conda `base` with pip-installed Torch if possible.

```bash
# From the project root.
cd fin_graph_rag_project

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
.../fin_graph_rag_project/.venv/bin/fin-graph-rag
```

If the CLI points to `/opt/anaconda3/bin/fin-graph-rag`, reinstall inside the active venv and clear the shell cache:

```bash
python -m pip install -e '.[lora,neo4j]'
hash -r
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

The project assumes your fine-tuned reranker adapter is local:

```text
fin_graph_rag_project/adaptor/
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

A valid reranker adapter should include a score/classifier head, for example:

```text
base_model.model.score.weight
```

If you see `SafetensorError: header too large`, the file is probably a Git LFS pointer or corrupted download. Fetch only the adapter folder with sparse clone:

```bash
cd ~/Downloads
brew install git-lfs  # skip if already installed
git lfs install

GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse <GITHUB_REPO_URL> ftRAG_adapter_only
cd ftRAG_adapter_only

git sparse-checkout set data/outputs/llama3_lora_reranker_quick/adapter
git lfs pull -I "data/outputs/llama3_lora_reranker_quick/adapter/**"

# Copy the real adapter into this project.
cd ~/Downloads/fin_graph_rag_project\ 2
rm -rf ./adaptor
cp -R ~/Downloads/ftRAG_adapter_only/data/outputs/llama3_lora_reranker_quick/adapter ./adaptor
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

Do **not** run full LLM claim extraction over all blocks unless you expect a long job. Use it only for a small demo subset:

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

This does not reparse PDFs. It indexes the same `Block.text` rows from SQLite.

```bash
fin-graph-rag build-indexes \
  --db artifacts/graph.sqlite \
  --index-dir artifacts/index
```

---

## 5. Optional: mirror SQLite graph to Neo4j for visualization

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

### 5.2 Start a second Neo4j container if port 7474/7687 is already in use

```bash
docker run --rm --name fin-neo4j-alt \
  -p 7475:7474 \
  -p 7688:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

Open:

```text
http://localhost:7475
```

Connect with:

```text
Connection URL: neo4j://localhost:7688
user: neo4j
password: password
```

### 5.3 Sync an existing SQLite graph to Neo4j

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

## 6. Retrieval smoke tests

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

## 7. Required accuracy/latency model variants

These are the five variants used for the slide-style accuracy/latency plot. Keep the answer model fixed across variants so the comparison isolates retrieval/reranking behavior.

| Slide label | CLI variant | Description |
|---|---|---|
| Dense | `dense` | FAISS dense retrieval only |
| Dense + BM25 | `dense_bm25` | Dense + BM25 reciprocal-rank fusion |
| Dense + BM25 + CE rerank | `dense_bm25_ce` | Generic cross-encoder reranking |
| Dense + BM25 + fine-tuned rerank | `dense_bm25_lora` | LoRA sequence-classification reranking |
| Dense + Graph | `dense_graph` | Dense seeds plus structural/semantic graph expansion |

Extra ablations such as `graph_hybrid`, `hybrid_lora`, and `graph_hybrid_lora` still exist, but they are not needed for the five-point slide.

Set CE and LoRA options:

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

### 7.2 Run 100-question evaluation

The cache reuses the first 50 rows if the settings and cache directory are unchanged.

```bash
for v in dense dense_bm25 dense_bm25_ce dense_bm25_lora dense_graph; do
  fin-graph-rag eval-financebench \
    --variant "$v" \
    --max-questions 100 \
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
    --out-dir "artifacts/results/e2e_${v}_100q" \
    --use-eval-cache \
    --eval-cache-dir artifacts/results/cache
 done
```

### 7.3 Run full 150-question evaluation

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

Shell note: the backslash must be the **last character** on each line. Do not put spaces after `\`.

### 7.4 Plot accuracy vs latency

50-question plot:

```bash
fin-graph-rag plot-latency \
  --result-dirs artifacts/results/e2e_dense_50q \
  --result-dirs artifacts/results/e2e_dense_bm25_50q \
  --result-dirs artifacts/results/e2e_dense_bm25_ce_50q \
  --result-dirs artifacts/results/e2e_dense_bm25_lora_50q \
  --result-dirs artifacts/results/e2e_dense_graph_50q \
  --out artifacts/results/accuracy_latency_50q.png
```

100-question plot:

```bash
fin-graph-rag plot-latency \
  --result-dirs artifacts/results/e2e_dense_100q \
  --result-dirs artifacts/results/e2e_dense_bm25_100q \
  --result-dirs artifacts/results/e2e_dense_bm25_ce_100q \
  --result-dirs artifacts/results/e2e_dense_bm25_lora_100q \
  --result-dirs artifacts/results/e2e_dense_graph_100q \
  --out artifacts/results/accuracy_latency_100q.png
```

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

---

## 8. Eval cache

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

## 9. LLM-as-judge risk/P&L causality evaluation

Set a real LLM endpoint. Without `OPENAI_API_KEY`, the project falls back to heuristics and should not be used for final judge metrics.

```bash
export OPENAI_API_KEY="YOUR_KEY"
export ANSWER_MODEL="gpt-4o-mini"
export JUDGE_MODEL="gpt-4o-mini"
```

Run the 2 persona × 2 task × 2 question judge setup:

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

Get the headline metric:

```bash
cat artifacts/results/persona_judge_2x2/summary.json
```

Metric definition:

```text
graph_hybrid_preference_rate = graph_hybrid wins / total judged comparisons
```

Example:

```text
6 / 8 = 0.75 = 75%
```

Inspect all 8 individual judgments:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("artifacts/results/persona_judge_2x2/comparisons.csv")
print(df[["persona", "task", "question", "preferred_variant", "score_a", "score_b", "rationale"]].to_string(index=False))

wins = (df["preferred_variant"] == "graph_hybrid").sum()
total = len(df)
print(f"\nGraph wins: {wins}/{total} = {wins/total:.1%}")
print("\nCounts:")
print(df["preferred_variant"].value_counts())
PY
```

Create a slide-style 8-cell preference table:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("artifacts/results/persona_judge_2x2/comparisons.csv")
df["q_num"] = df.groupby(["persona", "task"]).cumcount() + 1
df["winner"] = df["preferred_variant"].map({
    "graph_hybrid": "Dense + Graph",
    "hybrid": "Dense",
    "dense": "Dense",
    "tie": "Tie",
}).fillna(df["preferred_variant"])

table = df.pivot_table(
    index="persona",
    columns=["task", "q_num"],
    values="winner",
    aggfunc="first",
)

print(table)
table.to_csv("artifacts/results/persona_judge_2x2/preference_table_8_cells.csv")
print("\nWrote artifacts/results/persona_judge_2x2/preference_table_8_cells.csv")
PY
```

Note: `preference_matrix.csv` has 4 rows because it groups `2 personas × 2 tasks`; each row aggregates 2 questions. The raw 8-question file is `comparisons.csv`.

---

## 10. How GraphRAG expansion works

The implementation uses bounded one-hop expansion from initially retrieved seed blocks. It expands in multiple structural and semantic directions:

```text
seed Block
  ├─ same page:       Page <-HAS_BLOCK- other Blocks
  ├─ nearby blocks:   PREV_BLOCK / NEXT_BLOCK
  ├─ section context: Section <-IN_SECTION- other Blocks
  ├─ claims:          Block -> CausalClaim
  ├─ risk entities:   CausalClaim -> RiskEntity
  └─ P&L drivers:     CausalClaim -> PnlDriver
```

This is intentionally shallow. It avoids graph blow-up while recovering nearby table rows, section context, and risk/P&L causal evidence.

Retrieval flow:

```text
query
  -> dense/BM25 seeds
  -> one-hop structural + semantic graph expansion
  -> weighted merge
  -> optional CE or LoRA rerank
  -> final top-k evidence blocks
```

---

## 11. Common troubleshooting

### `ModuleNotFoundError: No module named 'fin_graph_rag'`

You are probably running an old CLI outside the project venv.

```bash
ls
# should show: README.md pyproject.toml src/ data/ configs/

python -m pip install -e '.[lora,neo4j]'
hash -r
which fin-graph-rag
```

Expected:

```text
.../fin_graph_rag_project/.venv/bin/fin-graph-rag
```

### Neo4j connection refused on `localhost:7687`

Check actual Docker port mapping:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

If you see:

```text
0.0.0.0:7688->7687/tcp
```

use:

```bash
--neo4j-uri bolt://127.0.0.1:7688
```

### Neo4j Browser cannot connect

Browser URL and database connection port are different:

```text
Open Browser UI:     http://localhost:7475
Connect to database: neo4j://localhost:7688
```

### LoRA warning: `score.weight | MISSING`

This is expected when loading an instruction base model as `LlamaForSequenceClassification`. It is OK if your adapter contains:

```text
base_model.model.score.weight
```

Validate:

```bash
python - <<'PY'
from safetensors import safe_open
with safe_open("./adaptor/adapter_model.safetensors", framework="pt", device="cpu") as f:
    print([k for k in f.keys() if "score" in k.lower() or "classifier" in k.lower()])
PY
```

### LoRA is too slow

Use a small rerank pool and shorter input:

```bash
--rerank-top-n 12 \
--top-k 8 \
--lora-max-length 256 \
--lora-batch-size 1
```

Do not call `fin-graph-rag retrieve` 150 times. Use one `eval-financebench` call so the model loads once for the whole evaluation.

### Retrieval misses FinanceBench numeric/table questions

Some FinanceBench terms need expansion, for example:

```text
PPNE -> property, plant and equipment — net
FY2018 -> year ended December 31, 2018
balance sheet -> consolidated balance sheet
```

If a single question fails, compare base retrieval and LoRA retrieval:

```bash
fin-graph-rag retrieve \
  "What is the year end FY2018 net PPNE for 3M?" \
  --db artifacts/graph.sqlite \
  --index-dir artifacts/index \
  --variant dense_bm25 \
  --doc-hint 3M_2018_10K \
  --doc-scope strict \
  --top-k 8
```

```bash
fin-graph-rag retrieve \
  "What is the year end FY2018 net PPNE for 3M?" \
  --db artifacts/graph.sqlite \
  --index-dir artifacts/index \
  --variant dense_bm25_lora \
  --doc-hint 3M_2018_10K \
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

---

## 12. Recommended final reporting

Use these outputs for the final report or slides:

```text
artifacts/results/accuracy_latency_150q.png
artifacts/results/persona_judge_2x2/summary.json
artifacts/results/persona_judge_2x2/comparisons.csv
artifacts/results/persona_judge_2x2/preference_table_8_cells.csv
```

Suggested wording for the judge metric:

```text
LLM-as-judge preferred Dense + Graph in 6 of 8 risk/P&L causality comparisons, suggesting graph expansion helps when questions require linking risk drivers to financial impact rather than extracting a single numeric field.
```
