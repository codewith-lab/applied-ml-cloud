from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from fin_graph_rag.generation.llm_client import LLMClient


@dataclass
class CorrectnessJudgment:
    is_correct: bool
    score: float
    rationale: str
    raw: str | None = None


@dataclass
class PairwiseJudgment:
    winner: str
    score_a: float
    score_b: float
    rationale: str
    raw: str | None = None


class Judge:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient.from_env()

    def judge_correctness(self, question: str, predicted: str, gold: str | None) -> CorrectnessJudgment:
        if not gold:
            return CorrectnessJudgment(False, 0.0, "No gold answer was available.")
        if self.llm.available:
            return self._llm_correctness(question, predicted, gold)
        return self._heuristic_correctness(predicted, gold)

    def pairwise(self, persona: str, task: str, question: str, answer_a: str, answer_b: str) -> PairwiseJudgment:
        if self.llm.available:
            return self._llm_pairwise(persona, task, question, answer_a, answer_b)
        # Fallback is deterministic and intentionally conservative.
        a_mentions = _risk_pnl_mentions(answer_a)
        b_mentions = _risk_pnl_mentions(answer_b)
        if b_mentions > a_mentions:
            winner = "B"
        elif a_mentions > b_mentions:
            winner = "A"
        else:
            winner = "tie"
        return PairwiseJudgment(winner=winner, score_a=float(a_mentions), score_b=float(b_mentions), rationale="Heuristic risk/P&L term coverage fallback.")

    def _llm_correctness(self, question: str, predicted: str, gold: str) -> CorrectnessJudgment:
        prompt = f"""
You are grading a financial QA answer. Decide whether the predicted answer is correct against the gold answer.
Accept equivalent phrasing, rounded numbers, and answers with extra context. Reject contradictions and unsupported claims.
Return strict JSON: {{"is_correct": boolean, "score": number between 0 and 1, "rationale": string}}

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}
""".strip()
        raw = self.llm.complete(prompt, model_role="judge", temperature=0)
        try:
            data = _json_from_text(raw)
            return CorrectnessJudgment(
                is_correct=bool(data.get("is_correct")),
                score=float(data.get("score", 0.0)),
                rationale=str(data.get("rationale", "")),
                raw=raw,
            )
        except Exception:
            fallback = self._heuristic_correctness(predicted, gold)
            fallback.raw = raw
            return fallback

    def _llm_pairwise(self, persona: str, task: str, question: str, answer_a: str, answer_b: str) -> PairwiseJudgment:
        prompt = f"""
You are a finance-domain judge comparing two RAG answers.
Persona: {persona}
Task: {task}
Question: {question}

Judge only on risk -> P&L causal reasoning quality:
- Does the answer identify the relevant risk driver?
- Does it link the driver to a concrete P&L line or financial effect?
- Is the answer grounded, specific, and non-hallucinated?
Do not reward pure numeric extraction unless it supports the causal chain.

Answer A:
{answer_a}

Answer B:
{answer_b}

Return strict JSON: {{"winner": "A|B|tie", "score_a": number, "score_b": number, "rationale": string}}
""".strip()
        raw = self.llm.complete(prompt, model_role="judge", temperature=0)
        data = _json_from_text(raw)
        winner = str(data.get("winner", "tie")).strip()
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        return PairwiseJudgment(
            winner=winner,
            score_a=float(data.get("score_a", 0.0)),
            score_b=float(data.get("score_b", 0.0)),
            rationale=str(data.get("rationale", "")),
            raw=raw,
        )

    @staticmethod
    def _heuristic_correctness(predicted: str, gold: str) -> CorrectnessJudgment:
        pred_nums = _numbers(predicted)
        gold_nums = _numbers(gold)
        if gold_nums and pred_nums:
            overlap = len(set(pred_nums) & set(gold_nums)) / max(1, len(set(gold_nums)))
            text_score = fuzz.token_set_ratio(predicted, gold) / 100.0
            score = max(overlap, text_score * 0.7)
        else:
            score = fuzz.token_set_ratio(predicted, gold) / 100.0
        return CorrectnessJudgment(score >= 0.72, float(score), "Fuzzy/numeric heuristic fallback.")


def _json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", text or "")


def _risk_pnl_mentions(text: str) -> int:
    low = (text or "").lower()
    risk_terms = ["risk", "cost", "currency", "supply", "demand", "regulation", "inflation", "interest", "tax"]
    pnl_terms = ["revenue", "sales", "margin", "expense", "income", "earnings", "cash flow", "profit", "loss"]
    return sum(t in low for t in risk_terms) + sum(t in low for t in pnl_terms)
