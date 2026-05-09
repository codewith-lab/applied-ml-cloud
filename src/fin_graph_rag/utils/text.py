from __future__ import annotations

import hashlib
import re
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9$%\.\-]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def stable_hash(*parts: object, length: int = 16) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def normalize_ws(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def sentence_split(text: str) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def estimate_tokens(text: str) -> int:
    # Conservative approximation for English financial filings.
    return max(1, int(len(text) / 4))


def chunk_text(text: str, max_chars: int) -> list[str]:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    buf = ""
    for sent in sentence_split(text):
        if len(buf) + len(sent) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        chunks.append(buf.strip())
    return chunks


def first_non_empty(values: Iterable[object]) -> str | None:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None
