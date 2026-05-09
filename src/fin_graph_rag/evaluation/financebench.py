from __future__ import annotations

import random
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from fin_graph_rag.utils.text import first_non_empty


def load_financebench(
    dataset_name: str,
    split: str,
    max_questions: Optional[int] = None,
    sample_questions: Optional[int] = None,
    sample_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load FinanceBench rows with optional deterministic random sampling.

    Semantics:
    - Load the full split.
    - Attach `_dataset_original_index` for reproducibility/debugging.
    - If `max_questions` is set, first restrict to the first N rows.
    - If `sample_questions` is set, draw a deterministic random sample from
      that candidate pool using `sample_seed`.

    Example: `--max-questions 150 --sample-questions 50` samples 50 from
    the first 150 rows. Returned rows are sorted by original dataset order so
    result CSVs remain stable and readable.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(ds):
        row = dict(r)
        row["_dataset_original_index"] = i
        rows.append(row)

    if max_questions is not None:
        rows = rows[:max_questions]

    if sample_questions is not None:
        if sample_questions < 0:
            raise ValueError("sample_questions must be non-negative")
        if sample_questions < len(rows):
            rng = random.Random(sample_seed)
            sampled = rng.sample(rows, sample_questions)
            rows = sorted(sampled, key=lambda r: int(r.get("_dataset_original_index", 0)))

    return rows


def get_question(row: dict[str, Any]) -> str:
    q = first_non_empty([row.get("question"), row.get("query"), row.get("Question"), row.get("prompt")])
    if not q:
        raise KeyError(f"Could not find question field in row keys: {sorted(row.keys())}")
    return q


def get_gold_answer(row: dict[str, Any]) -> str | None:
    return first_non_empty(
        [
            row.get("answer"),
            row.get("gold_answer"),
            row.get("final_answer"),
            row.get("Answer"),
            row.get("evidence_answer"),
        ]
    )


DOC_HINT_KEYS = {
    "doc",
    "doc_name",
    "doc_link",
    "doc_url",
    "document",
    "documents",
    "document_name",
    "file",
    "files",
    "filename",
    "file_name",
    "source",
    "sources",
    "evidence",
    "filing",
    "filing_name",
    "filing_url",
    "company",
    "company_name",
    "ticker",
    "symbol",
    "cik",
    "doc_type",
    "doc_period",
    "fiscal_year",
    "year",
}


def _collect_doc_hints(value: Any, parent_key: str | None = None) -> list[str]:
    hints: list[str] = []
    key = (parent_key or "").lower()
    key_is_doc_like = key in DOC_HINT_KEYS or any(
        token in key for token in ["doc", "file", "source", "evidence", "filing", "company", "ticker"]
    )

    if isinstance(value, dict):
        for k, v in value.items():
            hints.extend(_collect_doc_hints(v, str(k)))
    elif isinstance(value, list):
        for item in value:
            hints.extend(_collect_doc_hints(item, parent_key))
    elif value is not None and key_is_doc_like:
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            hints.append(text)
            hints.extend(_url_doc_hints(text))
    return hints


def _url_doc_hints(value: str) -> list[str]:
    """Return deterministic matching hints from a FinanceBench document URL.

    FinanceBench often provides `doc_link` values pointing to SEC or company
    filing URLs. The local corpus usually stores friendlier names such as
    `3M_2018_10K.pdf`, so the raw URL alone is not enough. These URL-derived
    fragments are still useful when the local file name contains an accession
    number or original PDF basename.
    """
    text = str(value or "").strip()
    if not re.match(r"https?://", text, flags=re.I):
        return []
    parsed = urlparse(text)
    basename = Path(parsed.path).name
    stem = Path(basename).stem if basename else ""
    parts = [basename, stem]
    # SEC accession-like tokens in URLs, e.g. 0001558370-19-000470.
    parts.extend(re.findall(r"\d{7,10}-\d{2}-\d{4,8}", text))
    return [p for p in parts if p]


def _first_by_keys(row: dict[str, Any], keys: set[str]) -> str | None:
    for k, v in row.items():
        lk = str(k).lower()
        if lk in keys or any(token in lk for token in keys):
            if v is not None and not isinstance(v, (dict, list)):
                text = str(v).strip()
                if text and text.lower() not in {"nan", "none", "null"}:
                    return text
    return None


def _normalize_doc_type(value: str | None) -> str | None:
    if not value:
        return None
    norm = re.sub(r"[^a-z0-9]+", "", value.lower())
    if norm in {"10k", "form10k", "annualreport"}:
        return "10K"
    if norm in {"10q", "form10q", "quarterlyreport"}:
        return "10Q"
    if norm in {"8k", "form8k"}:
        return "8K"
    return value.upper()


def _year_hint(value: str | int | None) -> str | None:
    if value is None:
        return None
    m = re.search(r"(?:19|20)\d{2}", str(value))
    return m.group(0) if m else None


def _structured_doc_hints(row: dict[str, Any]) -> list[str]:
    """Build filename-like hints from FinanceBench metadata.

    Example row fields such as `company=3M`, `doc_period=2018`, and
    `doc_type=10k` should match local PDFs named `3M_2018_10K.pdf`. This is
    more reliable than matching the raw `doc_link`, whose basename is often an
    SEC accession number.
    """
    company = _first_by_keys(row, {"company", "company_name", "entity", "issuer"})
    ticker = _first_by_keys(row, {"ticker", "symbol"})
    period = _year_hint(_first_by_keys(row, {"doc_period", "period", "fiscal_year", "year"}))
    doc_type = _normalize_doc_type(_first_by_keys(row, {"doc_type", "document_type", "filing_type", "form_type"}))

    names = [x for x in [company, ticker] if x]
    hints: list[str] = []
    for name in names:
        hints.append(name)
        if period:
            hints.append(f"{name}_{period}")
            hints.append(f"{name} {period}")
        if doc_type:
            hints.append(f"{name}_{doc_type}")
            hints.append(f"{name} {doc_type}")
        if period and doc_type:
            hints.extend([
                f"{name}_{period}_{doc_type}",
                f"{name} {period} {doc_type}",
                f"{name}_{doc_type}_{period}",
                f"{name} {doc_type} {period}",
            ])
    if period and doc_type:
        hints.append(f"{period}_{doc_type}")
        hints.append(f"{period} {doc_type}")
    return hints


def get_doc_hints(row: dict[str, Any]) -> list[str]:
    """Extract document hints for document-scoped FinanceBench retrieval.

    The open-source FinanceBench rows have used slightly different schemas
    across exports. This function looks for common top-level fields and nested
    metadata/evidence/source records so evaluation can restrict retrieval to the
    evidence filing instead of searching unrelated documents.
    """
    hints: list[str] = []
    # Start with structured metadata because it maps directly to the local
    # FinanceBench PDF naming convention, e.g. 3M_2018_10K.pdf.
    hints.extend(_structured_doc_hints(row))
    for key, value in row.items():
        hints.extend(_collect_doc_hints(value, key))

    # Preserve order while deduplicating.
    seen: set[str] = set()
    out: list[str] = []
    for h in hints:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out
