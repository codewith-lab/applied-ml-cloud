from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from fin_graph_rag.config import load_config
from fin_graph_rag.evaluation.e2e import run_financebench_eval
from fin_graph_rag.evaluation.persona_judge import run_persona_judge_eval
from fin_graph_rag.evaluation.plots import plot_accuracy_latency_from_summary
from fin_graph_rag.indexing.build_indexes import build_indexes
from fin_graph_rag.ingest.build_graph import build_graph, sync_neo4j_from_sqlite
from fin_graph_rag.retrieval.pipeline import RetrievalPipeline

app = typer.Typer(add_completion=False, help="Local finance GraphRAG pipeline")
console = Console()


@app.command()
def ingest(
    pdf_dir: str = typer.Option("data/pdfs", help="Directory containing evaluation PDFs."),
    db: str = typer.Option("artifacts/graph.sqlite", help="SQLite graph path."),
    config: Optional[str] = typer.Option(None, help="YAML config path."),
    use_llm_claims: bool = typer.Option(False, help="Use LLM for semantic claim extraction."),
    max_claim_blocks: Optional[int] = typer.Option(None, help="Optional cap for claim extraction smoke runs."),
    no_reset: bool = typer.Option(False, help="Do not reset existing graph DB."),
    neo4j_uri: Optional[str] = typer.Option(None, help="Optional Neo4j URI, e.g. bolt://localhost:7687. If set, sync graph to Neo4j after SQLite ingest."),
    neo4j_user: Optional[str] = typer.Option(None, help="Neo4j user."),
    neo4j_password: Optional[str] = typer.Option(None, help="Neo4j password."),
    neo4j_database: Optional[str] = typer.Option(None, help="Optional Neo4j database name."),
    no_neo4j_reset: bool = typer.Option(False, help="Do not clear Neo4j before syncing."),
) -> None:
    cfg = load_config(config)
    cfg.paths.pdf_dir = pdf_dir
    cfg.paths.graph_db = db
    counts = build_graph(
        pdf_dir=pdf_dir,
        db_path=db,
        parse_config=cfg.parse,
        use_llm_claims=use_llm_claims,
        max_claim_blocks=max_claim_blocks,
        reset=not no_reset,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        neo4j_reset=not no_neo4j_reset,
    )
    console.print(counts)


@app.command("sync-neo4j")
def sync_neo4j_cmd(
    db: str = typer.Option("artifacts/graph.sqlite", help="SQLite graph path."),
    neo4j_uri: str = typer.Option(..., help="Neo4j URI, e.g. bolt://localhost:7687."),
    neo4j_user: str = typer.Option("neo4j", help="Neo4j user."),
    neo4j_password: str = typer.Option(..., help="Neo4j password."),
    neo4j_database: Optional[str] = typer.Option(None, help="Optional Neo4j database name."),
    no_reset: bool = typer.Option(False, help="Do not clear Neo4j before syncing."),
    batch_size: int = typer.Option(1000, help="Neo4j batch size."),
) -> None:
    counts = sync_neo4j_from_sqlite(
        db_path=db,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        reset=not no_reset,
        batch_size=batch_size,
    )
    console.print(counts)


@app.command("build-indexes")
def build_indexes_cmd(
    db: str = typer.Option("artifacts/graph.sqlite", help="SQLite graph path."),
    index_dir: str = typer.Option("artifacts/index", help="Index output directory."),
    config: Optional[str] = typer.Option(None, help="YAML config path."),
    embedding_model: Optional[str] = typer.Option(None, help="SentenceTransformer model name."),
    batch_size: Optional[int] = typer.Option(None, help="Embedding batch size."),
) -> None:
    cfg = load_config(config)
    info = build_indexes(
        db,
        index_dir,
        embedding_model=embedding_model or cfg.embedding.model_name,
        batch_size=batch_size or cfg.embedding.batch_size,
    )
    console.print(info)


@app.command()
def retrieve(
    query: str = typer.Argument(...),
    db: str = typer.Option("artifacts/graph.sqlite", help="SQLite graph path."),
    index_dir: str = typer.Option("artifacts/index", help="Index directory."),
    config: Optional[str] = typer.Option(None, help="YAML config path."),
    variant: str = typer.Option("graph_hybrid", help="dense|dense_bm25|dense_bm25_ce|dense_bm25_lora|dense_graph|bm25|hybrid|hybrid_ce|hybrid_lora|graph_hybrid|graph_hybrid_lora"),
    top_k: int = typer.Option(8, help="Final number of blocks to show/use as evidence."),
    rerank_top_n: Optional[int] = typer.Option(12, help="Neural reranker candidate pool size. For speed, rerank 12 and return --top-k 8."),
    doc_hint: Optional[list[str]] = typer.Option(
        None,
        "--doc-hint",
        help="Restrict retrieval to matching ingested PDF(s). Repeatable, e.g. --doc-hint 3M --doc-hint 10K.",
    ),
    doc_scope: str = typer.Option(
        "fallback",
        help="Document scope for retrieval: fallback auto-infers a document from the query when possible; strict requires a matching --doc-hint; off searches the whole corpus.",
    ),
    lora_base_model: Optional[str] = typer.Option(None, help="Base model for existing LoRA adapter."),
    lora_adapter_path: Optional[str] = typer.Option(None, help="Existing LoRA adapter path."),
    lora_device: Optional[str] = typer.Option(None, help="LoRA device: cpu, cuda, mps, or auto. Default is cpu for macOS stability."),
    lora_dtype: Optional[str] = typer.Option(None, help="LoRA dtype: float32, float16, or bfloat16. Default is float32."),
    lora_batch_size: Optional[int] = typer.Option(None, help="LoRA scoring batch size. Use 1 for debugging segfaults/OOM."),
    lora_max_length: Optional[int] = typer.Option(None, help="LoRA tokenizer max length. Lower values reduce memory."),
    ce_model: Optional[str] = typer.Option(None, help="Cross-encoder model for CE rerank baseline."),
    ce_device: Optional[str] = typer.Option(None, help="Cross-encoder device: cpu, cuda, mps, or auto."),
    ce_batch_size: Optional[int] = typer.Option(None, help="Cross-encoder scoring batch size."),
    ce_max_length: Optional[int] = typer.Option(None, help="Cross-encoder tokenizer max length."),
) -> None:
    cfg = load_config(config)
    cfg.paths.graph_db = db
    cfg.paths.index_dir = index_dir
    variant_key = variant.lower().replace("-", "_")
    cfg.reranker.enabled = variant_key in {"graph_hybrid_lora", "hybrid_lora", "dense_bm25_lora"}
    cfg.cross_encoder.enabled = variant_key in {"hybrid_ce", "dense_bm25_ce", "graph_hybrid_ce"}
    if rerank_top_n:
        cfg.retrieval.rerank_top_n = rerank_top_n
    if lora_base_model:
        cfg.reranker.base_model = lora_base_model
    if lora_adapter_path:
        cfg.reranker.adapter_path = lora_adapter_path
    if lora_device:
        cfg.reranker.device = lora_device
    if lora_dtype:
        cfg.reranker.dtype = lora_dtype
    if lora_batch_size:
        cfg.reranker.batch_size = lora_batch_size
    if lora_max_length:
        cfg.reranker.max_length = lora_max_length
    if ce_model:
        cfg.cross_encoder.model_name = ce_model
    if ce_device:
        cfg.cross_encoder.device = ce_device
    if ce_batch_size:
        cfg.cross_encoder.batch_size = ce_batch_size
    if ce_max_length:
        cfg.cross_encoder.max_length = ce_max_length
    pipeline = RetrievalPipeline.load(db, index_dir, cfg.retrieval, cfg.reranker, cfg.cross_encoder)
    retrieved, latency_ms = pipeline.retrieve(query, variant=variant, top_k=top_k, doc_hints=doc_hint, doc_scope=doc_scope)
    pipeline.close()

    table = Table(title=f"{variant} retrieval, {latency_ms:.1f} ms")
    table.add_column("rank")
    table.add_column("score")
    table.add_column("source")
    table.add_column("file/page/block")
    table.add_column("text")
    for i, rb in enumerate(retrieved, start=1):
        b = rb.block
        table.add_row(
            str(i),
            f"{rb.score:.3f}",
            ",".join(rb.sources),
            f"{b.file_name} p.{b.page_num} b.{b.block_num}",
            b.text.replace("\n", " ")[:300],
        )
    console.print(table)


@app.command("eval-financebench")
def eval_financebench_cmd(
    dataset_name: Optional[str] = typer.Option(None, help="HF dataset name."),
    split: Optional[str] = typer.Option(None, help="Dataset split."),
    max_questions: Optional[int] = typer.Option(
        None,
        "--max-questions",
        "--question-limit",
        help="Restrict evaluation to the first N rows, e.g. 50, 100, or 150.",
    ),
    sample_questions: Optional[int] = typer.Option(None, help="Sample N from candidate pool."),
    sample_seed: int = typer.Option(42, help="Sampling seed."),
    doc_scope: str = typer.Option(
        "strict",
        help="FinanceBench document scope: strict restricts each question to its evidence PDF; fallback uses doc scope when matched; off searches the whole corpus.",
    ),
    variant: str = typer.Option("graph_hybrid", help="dense|dense_bm25|dense_bm25_ce|dense_bm25_lora|dense_graph|bm25|hybrid|hybrid_ce|hybrid_lora|graph_hybrid|graph_hybrid_lora"),
    top_k: Optional[int] = typer.Option(None, help="Final number of evidence blocks per question. Defaults to config retrieval.final_k."),
    rerank_top_n: Optional[int] = typer.Option(None, help="Neural reranker candidate pool size. Use 12 for fast LoRA eval."),
    db: str = typer.Option("artifacts/graph.sqlite", help="SQLite graph path."),
    index_dir: str = typer.Option("artifacts/index", help="Index directory."),
    out_dir: Optional[str] = typer.Option(None, help="Output directory."),
    config: Optional[str] = typer.Option(None, help="YAML config path."),
    lora_base_model: Optional[str] = typer.Option(None, help="Base model for existing LoRA adapter."),
    lora_adapter_path: Optional[str] = typer.Option(None, help="Existing LoRA adapter path."),
    lora_device: Optional[str] = typer.Option(None, help="LoRA device: cpu, cuda, mps, or auto. Default is cpu for macOS stability."),
    lora_dtype: Optional[str] = typer.Option(None, help="LoRA dtype: float32, float16, or bfloat16. Default is float32."),
    lora_batch_size: Optional[int] = typer.Option(None, help="LoRA scoring batch size. Use 1 for debugging segfaults/OOM."),
    lora_max_length: Optional[int] = typer.Option(None, help="LoRA tokenizer max length. Lower values reduce memory."),
    ce_model: Optional[str] = typer.Option(None, help="Cross-encoder model for CE rerank baseline."),
    ce_device: Optional[str] = typer.Option(None, help="Cross-encoder device: cpu, cuda, mps, or auto."),
    ce_batch_size: Optional[int] = typer.Option(None, help="Cross-encoder scoring batch size."),
    ce_max_length: Optional[int] = typer.Option(None, help="Cross-encoder tokenizer max length."),
    use_eval_cache: bool = typer.Option(False, "--use-eval-cache/--no-use-eval-cache", help="Cache per-question FinanceBench results across 25/50/100 sweeps."),
    eval_cache_dir: Optional[str] = typer.Option(None, help="Directory for per-question eval cache CSVs."),
) -> None:
    cfg = load_config(config)
    cfg.paths.graph_db = db
    cfg.paths.index_dir = index_dir
    if top_k is not None:
        cfg.retrieval.final_k = top_k
    if rerank_top_n is not None:
        cfg.retrieval.rerank_top_n = rerank_top_n
    variant_key = variant.lower().replace("-", "_")
    cfg.reranker.enabled = variant_key in {"graph_hybrid_lora", "hybrid_lora", "dense_bm25_lora"}
    cfg.cross_encoder.enabled = variant_key in {"hybrid_ce", "dense_bm25_ce", "graph_hybrid_ce"}
    if lora_base_model:
        cfg.reranker.base_model = lora_base_model
    if lora_adapter_path:
        cfg.reranker.adapter_path = lora_adapter_path
    if lora_device:
        cfg.reranker.device = lora_device
    if lora_dtype:
        cfg.reranker.dtype = lora_dtype
    if lora_batch_size:
        cfg.reranker.batch_size = lora_batch_size
    if lora_max_length:
        cfg.reranker.max_length = lora_max_length
    if ce_model:
        cfg.cross_encoder.model_name = ce_model
    if ce_device:
        cfg.cross_encoder.device = ce_device
    if ce_batch_size:
        cfg.cross_encoder.batch_size = ce_batch_size
    if ce_max_length:
        cfg.cross_encoder.max_length = ce_max_length
    run_financebench_eval(
        cfg,
        dataset_name=dataset_name,
        split=split,
        max_questions=max_questions,
        sample_questions=sample_questions,
        sample_seed=sample_seed,
        variant=variant,
        out_dir=out_dir,
        use_eval_cache=use_eval_cache,
        cache_dir=eval_cache_dir,
        doc_scope=doc_scope,
    )


@app.command("judge-personas")
def judge_personas_cmd(
    db: str = typer.Option("artifacts/graph.sqlite", help="SQLite graph path."),
    index_dir: str = typer.Option("artifacts/index", help="Index directory."),
    out_dir: Optional[str] = typer.Option(None, help="Output directory."),
    num_personas: int = typer.Option(2, help="Number of generated personas."),
    questions_per_task: int = typer.Option(2, help="Questions per task/persona."),
    config: Optional[str] = typer.Option(None, help="YAML config path."),
) -> None:
    cfg = load_config(config)
    cfg.paths.graph_db = db
    cfg.paths.index_dir = index_dir
    run_persona_judge_eval(cfg, num_personas=num_personas, questions_per_task=questions_per_task, out_dir=out_dir)


@app.command("plot-latency")
def plot_latency_cmd(
    result_dirs: list[str] = typer.Option(..., help="Result directories containing summary.csv."),
    out: str = typer.Option("artifacts/results/accuracy_latency_all.png", help="Output PNG path."),
) -> None:
    summary_files = [Path(d) / "summary.csv" for d in result_dirs]
    plot_accuracy_latency_from_summary(summary_files, out)
    console.print(f"Wrote {out}")


@app.command("show-summary")
def show_summary(result_dir: str = typer.Argument(...)) -> None:
    summary = pd.read_csv(Path(result_dir) / "summary.csv")
    console.print(summary)


if __name__ == "__main__":
    app()
