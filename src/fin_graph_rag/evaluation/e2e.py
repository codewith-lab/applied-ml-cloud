from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from rich.console import Console
from tqdm import tqdm

from fin_graph_rag.config import AppConfig, ensure_dir
from fin_graph_rag.evaluation.financebench import get_doc_hints, get_gold_answer, get_question, load_financebench
from fin_graph_rag.evaluation.judge import Judge
from fin_graph_rag.evaluation.plots import plot_accuracy_latency_from_summary
from fin_graph_rag.generation.answerer import Answerer
from fin_graph_rag.generation.llm_client import LLMClient
from fin_graph_rag.retrieval.pipeline import RetrievalPipeline

console = Console()


def _safe_cache_token(value: str | None) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown")
    return token.strip("_") or "unknown"


def _cache_path(
    cfg: AppConfig, dataset_name: str, split: str, variant: str, doc_scope: str, cache_dir: str | Path | None
) -> Path:
    root = ensure_dir(cache_dir or Path(cfg.paths.results_dir) / "cache")
    return root / f"financebench_{_safe_cache_token(dataset_name)}_{_safe_cache_token(split)}_{_safe_cache_token(variant)}_{_safe_cache_token(doc_scope)}.csv"


def _record_key(row_or_record: dict) -> str:
    value = row_or_record.get("_dataset_original_index", row_or_record.get("dataset_original_index"))
    return str(int(value)) if value is not None and str(value) != "nan" else str(row_or_record.get("question", ""))


def run_financebench_eval(
    cfg: AppConfig,
    dataset_name: str | None = None,
    split: str | None = None,
    max_questions: int | None = None,
    sample_questions: int | None = None,
    sample_seed: int | None = None,
    variant: str = "graph_hybrid",
    out_dir: str | Path | None = None,
    use_eval_cache: bool = False,
    cache_dir: str | Path | None = None,
    doc_scope: str | None = None,
) -> dict[str, float | str | int]:
    out = ensure_dir(out_dir or Path(cfg.paths.results_dir) / f"e2e_{variant}")
    dataset_name = dataset_name or cfg.evaluation.dataset_name
    split = split or cfg.evaluation.split
    max_questions = cfg.evaluation.max_questions if max_questions is None else max_questions
    sample_questions = cfg.evaluation.sample_questions if sample_questions is None else sample_questions
    sample_seed = cfg.evaluation.sample_seed if sample_seed is None else sample_seed
    doc_scope = (doc_scope or cfg.evaluation.doc_scope or "strict").lower()
    if doc_scope not in {"strict", "fallback", "off"}:
        raise ValueError("doc_scope must be one of: strict, fallback, off")

    rows = load_financebench(dataset_name, split, max_questions, sample_questions, sample_seed)
    console.print(f"Loaded {len(rows)} questions from {dataset_name}:{split}")

    cache_file: Path | None = None
    cached_by_key: dict[str, dict] = {}
    if use_eval_cache:
        cache_file = _cache_path(cfg, dataset_name, split, variant, doc_scope, cache_dir)
        if cache_file.exists():
            cache_df = pd.read_csv(cache_file)
            cached_by_key = {str(r["dataset_original_index"]): dict(r) for _, r in cache_df.iterrows() if pd.notna(r.get("dataset_original_index"))}
            console.print(f"Loaded {len(cached_by_key)} cached rows from {cache_file}")
        else:
            console.print(f"Eval cache enabled; will create {cache_file}")

    records_by_key: dict[str, dict] = {}
    rows_to_run = []
    for row in rows:
        key = _record_key(row)
        q = get_question(row)
        cached = cached_by_key.get(key)
        if (
            cached is not None
            and cached.get("variant") == variant
            and cached.get("question") == q
            and str(cached.get("doc_scope", doc_scope)) == doc_scope
        ):
            records_by_key[key] = cached
        else:
            rows_to_run.append(row)

    cache_hits = len(rows) - len(rows_to_run)
    if use_eval_cache:
        console.print(f"Cache hits: {cache_hits}; to evaluate: {len(rows_to_run)}")

    new_records = []
    if rows_to_run:
        pipeline = RetrievalPipeline.load(cfg.paths.graph_db, cfg.paths.index_dir, cfg.retrieval, cfg.reranker, cfg.cross_encoder)
        llm = LLMClient.from_env(cfg.llm.answer_model, cfg.llm.judge_model, cfg.llm.base_url, cfg.llm.timeout_seconds)
        answerer = Answerer(llm)
        judge = Judge(llm)

        for row in tqdm(rows_to_run, desc=f"eval {variant}"):
            q = get_question(row)
            gold = get_gold_answer(row)
            doc_hints = get_doc_hints(row)
            retrieved, retr_ms = pipeline.retrieve(q, variant=variant, doc_hints=doc_hints, doc_scope=doc_scope)
            answer = answerer.answer(q, retrieved, retrieval_latency_ms=retr_ms)
            judgment = judge.judge_correctness(q, answer.answer, gold)
            rec = {
                "dataset_name": dataset_name,
                "split": split,
                "dataset_original_index": row.get("_dataset_original_index"),
                "variant": variant,
                "doc_scope": doc_scope,
                "doc_hints": "|".join(doc_hints),
                "question": q,
                "gold_answer": gold,
                "predicted_answer": answer.answer,
                "is_correct": judgment.is_correct,
                "judge_score": judgment.score,
                "judge_rationale": judgment.rationale,
                "latency_ms": answer.latency_ms,
                "retrieval_latency_ms": answer.metadata.get("retrieval_latency_ms"),
                "num_retrieved": len(retrieved),
                "top_sources": ";".join(sorted({s for rb in retrieved for s in rb.sources})),
                "top_evidence": "\n".join(
                    f"{rb.block.file_name} p.{rb.block.page_num} b.{rb.block.block_num} score={rb.score:.3f}"
                    for rb in retrieved[:5]
                ),
            }
            key = _record_key(row)
            records_by_key[key] = rec
            new_records.append(rec)

        pipeline.close()

    if use_eval_cache and cache_file is not None and new_records:
        existing = pd.DataFrame(list(cached_by_key.values())) if cached_by_key else pd.DataFrame()
        updated = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
        updated = updated.drop_duplicates(subset=["dataset_original_index"], keep="last")
        updated = updated.sort_values("dataset_original_index")
        updated.to_csv(cache_file, index=False)
        console.print(f"Updated eval cache: {cache_file} ({len(updated)} rows)")

    ordered_records = [records_by_key[_record_key(row)] for row in rows]
    df = pd.DataFrame(ordered_records)
    df.to_csv(out / "per_question.csv", index=False)
    summary = {
        "variant": variant,
        "doc_scope": doc_scope,
        "n": int(len(df)),
        "accuracy": float(df["is_correct"].mean()) if len(df) else 0.0,
        "judge_score_mean": float(df["judge_score"].mean()) if len(df) else 0.0,
        "latency_ms_mean": float(df["latency_ms"].mean()) if len(df) else 0.0,
        "latency_ms_p50": float(df["latency_ms"].quantile(0.5)) if len(df) else 0.0,
        "latency_ms_p95": float(df["latency_ms"].quantile(0.95)) if len(df) else 0.0,
        "retrieval_latency_ms_mean": float(df["retrieval_latency_ms"].mean()) if len(df) else 0.0,
    }
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    plot_accuracy_latency_from_summary([out / "summary.csv"], out / "accuracy_latency.png")
    console.print(summary)
    return summary
