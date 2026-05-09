from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from tqdm import tqdm

from fin_graph_rag.config import AppConfig, ensure_dir
from fin_graph_rag.evaluation.judge import Judge
from fin_graph_rag.generation.answerer import Answerer
from fin_graph_rag.generation.llm_client import LLMClient
from fin_graph_rag.ingest.graph_store import GraphStore
from fin_graph_rag.retrieval.pipeline import RetrievalPipeline

console = Console()

RISK_PNL_TASKS = [
    "Risk driver identification: identify the business or market risk that could drive a financial result.",
    "PnL causal impact attribution: explain how a risk or operating driver affects revenue, margin, expenses, income, or cash flow.",
]

FALLBACK_PERSONAS = [
    "Portfolio manager evaluating downside risks and forward-looking P&L impact.",
    "Risk analyst mapping disclosed risk factors to concrete financial statement drivers.",
]

FALLBACK_QUESTIONS = [
    "Which disclosed risks could pressure gross margin, and what evidence links those risks to cost or pricing drivers?",
    "What operating or market risk appears most likely to affect revenue growth, and through what causal channel?",
    "How could foreign currency, interest rates, or inflation affect earnings or cash flow?",
    "Which risk factor has a plausible causal impact on expenses or operating income?",
]


def run_persona_judge_eval(
    cfg: AppConfig,
    num_personas: int = 2,
    questions_per_task: int = 2,
    out_dir: str | Path | None = None,
) -> dict[str, str | int | float]:
    out = ensure_dir(out_dir or Path(cfg.paths.results_dir) / "persona_judge")
    llm = LLMClient.from_env(cfg.llm.answer_model, cfg.llm.judge_model, cfg.llm.base_url, cfg.llm.timeout_seconds)
    judge = Judge(llm)
    answerer = Answerer(llm)
    pipeline = RetrievalPipeline.load(cfg.paths.graph_db, cfg.paths.index_dir, cfg.retrieval, cfg.reranker, cfg.cross_encoder)

    corpus_summary = summarize_corpus(cfg.paths.graph_db)
    personas = generate_personas(llm, corpus_summary, num_personas)
    questions = generate_questions(llm, corpus_summary, personas, questions_per_task)

    (out / "personas.json").write_text(json.dumps(personas, indent=2), encoding="utf-8")
    with (out / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    rows = []
    for item in tqdm(questions, desc="persona judge"):
        question = item["question"]
        persona = item["persona"]
        task = item["task"]
        ret_a, ms_a = pipeline.retrieve(question, variant="hybrid")
        ret_b, ms_b = pipeline.retrieve(question, variant="graph_hybrid")
        ans_a = answerer.answer(question, ret_a, retrieval_latency_ms=ms_a)
        ans_b = answerer.answer(question, ret_b, retrieval_latency_ms=ms_b)
        pj = judge.pairwise(persona, task, question, ans_a.answer, ans_b.answer)
        rows.append(
            {
                "persona": persona,
                "task": task,
                "question": question,
                "answer_a_variant": "hybrid",
                "answer_b_variant": "graph_hybrid",
                "winner": pj.winner,
                "preferred_variant": "hybrid" if pj.winner == "A" else ("graph_hybrid" if pj.winner == "B" else "tie"),
                "score_a": pj.score_a,
                "score_b": pj.score_b,
                "rationale": pj.rationale,
                "latency_a_ms": ans_a.latency_ms,
                "latency_b_ms": ans_b.latency_ms,
                "answer_a": ans_a.answer,
                "answer_b": ans_b.answer,
            }
        )

    pipeline.close()
    df = pd.DataFrame(rows)
    df.to_csv(out / "comparisons.csv", index=False)
    matrix = pd.crosstab([df["persona"], df["task"]], df["preferred_variant"])
    matrix.to_csv(out / "preference_matrix.csv")
    pref_rate = float((df["preferred_variant"] == "graph_hybrid").mean()) if len(df) else 0.0
    summary = {"n": len(df), "graph_hybrid_preference_rate": pref_rate, "out_dir": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(summary)
    return summary


def summarize_corpus(db_path: str | Path, max_examples: int = 40) -> str:
    store = GraphStore(db_path)
    docs = store.conn.execute("SELECT file_name, num_pages FROM documents ORDER BY file_name LIMIT 50").fetchall()
    claims = store.conn.execute(
        "SELECT claim_text, risk_entity, pnl_driver FROM claims ORDER BY confidence DESC LIMIT ?", (max_examples,)
    ).fetchall()
    store.close()
    doc_text = "; ".join(f"{d['file_name']} ({d['num_pages']} pages)" for d in docs[:20])
    claim_text = "\n".join(
        f"- risk={c['risk_entity']} -> pnl={c['pnl_driver']}: {c['claim_text'][:240]}" for c in claims
    )
    return f"Corpus documents: {doc_text}\n\nHigh-confidence risk/P&L claim examples:\n{claim_text}"


def generate_personas(llm: LLMClient, corpus_summary: str, num_personas: int) -> list[str]:
    if not llm.available:
        return FALLBACK_PERSONAS[:num_personas]
    prompt = f"""
Generate {num_personas} realistic finance user personas for asking questions over SEC filings.
Personas should care about risk -> P&L causal reasoning, not just number lookup.
Return strict JSON: {{"personas": [string, ...]}}

Corpus summary:
{corpus_summary[:5000]}
""".strip()
    raw = llm.complete(prompt, model_role="judge", temperature=0.2)
    try:
        data = json.loads(raw)
        personas = [str(p) for p in data.get("personas", []) if str(p).strip()]
        return (personas + FALLBACK_PERSONAS)[:num_personas]
    except Exception:
        return FALLBACK_PERSONAS[:num_personas]


def generate_questions(
    llm: LLMClient,
    corpus_summary: str,
    personas: list[str],
    questions_per_task: int,
) -> list[dict[str, str]]:
    if not llm.available:
        out = []
        i = 0
        for persona in personas:
            for task in RISK_PNL_TASKS:
                for _ in range(questions_per_task):
                    out.append({"persona": persona, "task": task, "question": FALLBACK_QUESTIONS[i % len(FALLBACK_QUESTIONS)]})
                    i += 1
        return out

    all_items: list[dict[str, str]] = []
    for persona in personas:
        for task in RISK_PNL_TASKS:
            prompt = f"""
Generate {questions_per_task} answerable, high-level questions for the persona and task below.
Questions must require risk -> P&L causal linkage. Avoid pure numeric extraction.
Return strict JSON: {{"questions": [string, ...]}}

Persona: {persona}
Task: {task}
Corpus summary:
{corpus_summary[:5000]}
""".strip()
            raw = llm.complete(prompt, model_role="judge", temperature=0.4)
            try:
                data = json.loads(raw)
                qs = [str(q) for q in data.get("questions", []) if str(q).strip()]
            except Exception:
                qs = FALLBACK_QUESTIONS[:questions_per_task]
            for q in (qs + FALLBACK_QUESTIONS)[:questions_per_task]:
                all_items.append({"persona": persona, "task": task, "question": q})
    return all_items
