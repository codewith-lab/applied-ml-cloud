from __future__ import annotations

import json
import re
from importlib import resources
from typing import Iterable

from tqdm import tqdm

from fin_graph_rag.generation.llm_client import LLMClient
from fin_graph_rag.models import BlockRecord, CausalClaim
from fin_graph_rag.utils.text import sentence_split, stable_hash

RISK_TERMS = [
    "risk",
    "supply chain",
    "foreign currency",
    "currency",
    "inflation",
    "interest rate",
    "tax",
    "tariff",
    "regulation",
    "competition",
    "acquisition",
    "restructuring",
    "impairment",
    "credit",
    "liquidity",
    "demand",
    "pricing",
    "commodity",
    "freight",
    "labor",
    "inventory",
    "brexit",
    "pandemic",
]

PNL_TERMS = [
    "revenue",
    "sales",
    "gross margin",
    "margin",
    "cost of goods sold",
    "cogs",
    "expense",
    "expenses",
    "operating income",
    "net income",
    "earnings",
    "ebitda",
    "cash flow",
    "profit",
    "loss",
    "fees",
]

DIRECTION_TERMS = {
    "increase": "increases",
    "increased": "increases",
    "higher": "increases",
    "raise": "increases",
    "pressure": "decreases",
    "pressures": "decreases",
    "decrease": "decreases",
    "decreased": "decreases",
    "lower": "decreases",
    "reduce": "decreases",
    "reduced": "decreases",
    "decline": "decreases",
    "declined": "decreases",
    "adverse": "decreases",
    "negative": "decreases",
    "unfavorable": "decreases",
    "impact": "impacts",
    "impacted": "impacts",
    "affect": "impacts",
    "affected": "impacts",
}


def load_claim_extraction_prompt() -> str:
    return resources.files("fin_graph_rag.prompts").joinpath("claim_extraction.txt").read_text(encoding="utf-8")


class CausalClaimExtractor:
    def __init__(
        self,
        use_llm: bool = False,
        llm_client: LLMClient | None = None,
        min_confidence: float = 0.35,
        prompt_template: str | None = None,
        show_progress: bool = True,
    ) -> None:
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.prompt_template = prompt_template or load_claim_extraction_prompt()
        self.show_progress = show_progress

    def extract_many(self, blocks: Iterable[BlockRecord], max_blocks: int | None = None) -> list[CausalClaim]:
        claims: list[CausalClaim] = []
        iterator = blocks
        if max_blocks is not None:
            # tqdm total is exact when a smoke cap is used.
            iterator = _take(blocks, max_blocks)
            total = max_blocks
        else:
            total = len(blocks) if hasattr(blocks, "__len__") else None
        if self.show_progress:
            mode = "LLM" if self.use_llm and self.llm_client and self.llm_client.available else "regex"
            iterator = tqdm(iterator, total=total, desc=f"claim blocks ({mode})", unit="block")
        for block in iterator:
            if self.use_llm and self.llm_client and self.llm_client.available:
                claims.extend(self._extract_with_llm(block))
            else:
                claims.extend(self._extract_with_regex(block))
        return [c for c in claims if c.confidence >= self.min_confidence]

    def _extract_with_regex(self, block: BlockRecord) -> list[CausalClaim]:
        out: list[CausalClaim] = []
        text_lower = block.text.lower()
        if not any(t in text_lower for t in RISK_TERMS) or not any(t in text_lower for t in PNL_TERMS):
            return out
        for sent in sentence_split(block.text):
            low = sent.lower()
            risk = self._first_match(low, RISK_TERMS)
            pnl = self._first_match(low, PNL_TERMS)
            direction = self._direction(low)
            if not (risk and pnl and direction):
                continue
            confidence = 0.45
            if "because" in low or "due to" in low or "result" in low or "driven by" in low:
                confidence += 0.2
            if block.block_type in {"paragraph", "table"}:
                confidence += 0.05
            claim_id = stable_hash(block.block_id, sent, risk, pnl)
            out.append(
                CausalClaim(
                    claim_id=claim_id,
                    source_block_id=block.block_id,
                    claim_text=sent[:1200],
                    risk_entity=risk,
                    pnl_driver=pnl,
                    direction=direction,
                    confidence=min(confidence, 0.95),
                    extractor="regex",
                )
            )
        return out

    def _extract_with_llm(self, block: BlockRecord) -> list[CausalClaim]:
        assert self.llm_client is not None
        prompt = self.prompt_template.format(
            file_name=block.file_name,
            page_num=block.page_num,
            block_type=block.block_type,
            section_title=block.section_title or "",
            block_text=block.text[:3500],
        )
        raw = self.llm_client.complete(prompt, model_role="answer", temperature=0)
        try:
            data = json.loads(_strip_json_fences(raw))
        except Exception:
            return self._extract_with_regex(block)
        out: list[CausalClaim] = []
        for item in data.get("claims", []):
            claim_text = str(item.get("claim_text", "")).strip()
            if not claim_text:
                continue
            risk = str(item.get("risk_entity", "")).strip() or None
            pnl = str(item.get("pnl_driver", "")).strip() or None
            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            out.append(
                CausalClaim(
                    claim_id=stable_hash(block.block_id, claim_text, risk, pnl),
                    source_block_id=block.block_id,
                    claim_text=claim_text[:1200],
                    risk_entity=risk,
                    pnl_driver=pnl,
                    direction=str(item.get("direction", "unknown")).strip().lower() or None,
                    confidence=confidence,
                    extractor="llm",
                )
            )
        return out

    @staticmethod
    def _first_match(text: str, terms: list[str]) -> str | None:
        for term in terms:
            if term in text:
                return term
        return None

    @staticmethod
    def _direction(text: str) -> str | None:
        for term, direction in DIRECTION_TERMS.items():
            if re.search(rf"\b{re.escape(term)}\b", text):
                return direction
        return None


def _take(blocks: Iterable[BlockRecord], n: int) -> Iterable[BlockRecord]:
    for i, block in enumerate(blocks):
        if i >= n:
            break
        yield block


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_query_terms(query: str) -> dict[str, list[str]]:
    low = query.lower()
    risks = [t for t in RISK_TERMS if t in low]
    pnls = [t for t in PNL_TERMS if t in low]
    directions = sorted({v for k, v in DIRECTION_TERMS.items() if re.search(rf"\b{re.escape(k)}\b", low)})
    return {"risk_terms": risks, "pnl_terms": pnls, "directions": directions}
