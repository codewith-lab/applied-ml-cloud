from __future__ import annotations

import os
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class LLMClient:
    answer_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    base_url: str | None = None
    timeout_seconds: int = 90

    @property
    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    @classmethod
    def from_env(
        cls,
        answer_model: str | None = None,
        judge_model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 90,
    ) -> "LLMClient":
        return cls(
            answer_model=answer_model or os.getenv("ANSWER_MODEL", "gpt-4o-mini"),
            judge_model=judge_model or os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout_seconds=timeout_seconds,
        )

    def complete(self, prompt: str, model_role: str = "answer", temperature: float = 0.0) -> str:
        if not self.available:
            raise RuntimeError("No OPENAI_API_KEY found. Set an OpenAI-compatible API key for LLM calls.")
        return self._complete_with_retry(prompt, model_role=model_role, temperature=temperature)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=12), stop=stop_after_attempt(3))
    def _complete_with_retry(self, prompt: str, model_role: str, temperature: float) -> str:
        from openai import OpenAI

        model = self.judge_model if model_role == "judge" else self.answer_model
        client = OpenAI(base_url=self.base_url, timeout=self.timeout_seconds) if self.base_url else OpenAI(timeout=self.timeout_seconds)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
