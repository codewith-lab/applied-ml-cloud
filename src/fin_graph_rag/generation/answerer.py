from __future__ import annotations

import time

from fin_graph_rag.generation.llm_client import LLMClient
from fin_graph_rag.models import AnswerResult, RetrievedBlock


class Answerer:
    def __init__(self, llm_client: LLMClient | None = None, model: str | None = None) -> None:
        self.llm = llm_client or LLMClient.from_env()
        self.model = model or self.llm.answer_model

    def answer(self, question: str, retrieved: list[RetrievedBlock], retrieval_latency_ms: float = 0.0) -> AnswerResult:
        start = time.perf_counter()
        if self.llm.available:
            answer = self._llm_answer(question, retrieved)
        else:
            answer = self._extractive_fallback(question, retrieved)
        total_latency = retrieval_latency_ms + (time.perf_counter() - start) * 1000
        return AnswerResult(
            question=question,
            answer=answer,
            retrieved=retrieved,
            latency_ms=total_latency,
            model=self.model,
            metadata={"retrieval_latency_ms": retrieval_latency_ms},
        )

    def _llm_answer(self, question: str, retrieved: list[RetrievedBlock]) -> str:
        context = self._format_context(retrieved)
        prompt = f"""
You are answering finance filing questions using only the supplied evidence.
Rules:
- Answer directly and concisely.
- Include the key number(s) or causal linkage when relevant.
- If evidence is insufficient, say exactly what is missing.
- Cite evidence as [file p.page block block_num].

Question: {question}

Evidence:
{context}

Answer:
""".strip()
        return self.llm.complete(prompt, model_role="answer", temperature=0)

    def _extractive_fallback(self, question: str, retrieved: list[RetrievedBlock]) -> str:
        if not retrieved:
            return "Insufficient evidence retrieved."
        snippets = []
        for rb in retrieved[:3]:
            b = rb.block
            text = b.text.replace("\n", " ")[:600]
            snippets.append(f"[{b.file_name} p.{b.page_num} block {b.block_num}] {text}")
        return "LLM not configured; top retrieved evidence:\n" + "\n".join(snippets)

    @staticmethod
    def _format_context(retrieved: list[RetrievedBlock]) -> str:
        parts = []
        for i, rb in enumerate(retrieved, start=1):
            b = rb.block
            parts.append(
                f"[{i}] [{b.file_name} p.{b.page_num} block {b.block_num} type={b.block_type} score={rb.score:.4f}]\n{b.text}"
            )
        return "\n\n".join(parts)
