from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

from fin_graph_rag.models import BlockRecord, DocumentRecord, PageRecord, TableRecord
from fin_graph_rag.utils.text import chunk_text, estimate_tokens, normalize_ws, stable_hash

_NUMERIC_RE = re.compile(r"(^|\s)[($-]?\d[\d,]*(\.\d+)?%?[)]?(\s|$)")
_TABLE_CAPTION_RE = re.compile(r"\b(table|schedule|consolidated|statement of)\b", re.IGNORECASE)


class PDFParser:
    def __init__(
        self,
        min_chars_per_block: int = 25,
        max_block_chars: int = 2500,
        merge_short_blocks: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.min_chars_per_block = min_chars_per_block
        self.max_block_chars = max_block_chars
        self.merge_short_blocks = merge_short_blocks
        self.show_progress = show_progress

    def parse_dir(
        self, pdf_dir: str | Path
    ) -> tuple[list[DocumentRecord], list[PageRecord], list[BlockRecord], list[TableRecord]]:
        paths = sorted(Path(pdf_dir).glob("*.pdf"))
        docs: list[DocumentRecord] = []
        pages: list[PageRecord] = []
        blocks: list[BlockRecord] = []
        tables: list[TableRecord] = []
        iterator = tqdm(paths, desc="parse PDFs", unit="pdf") if self.show_progress else paths
        for path in iterator:
            d, p, b, t = self.parse_pdf(path)
            docs.append(d)
            pages.extend(p)
            blocks.extend(b)
            tables.extend(t)
        return docs, pages, blocks, tables

    def parse_pdf(
        self, path: str | Path
    ) -> tuple[DocumentRecord, list[PageRecord], list[BlockRecord], list[TableRecord]]:
        path = Path(path)
        file_bytes = path.read_bytes()
        doc_sha = hashlib.sha1(file_bytes).hexdigest()
        doc_id = stable_hash(path.name, doc_sha)

        pdf = fitz.open(path)
        doc_record = DocumentRecord(
            doc_id=doc_id,
            file_name=path.name,
            path=str(path),
            sha1=doc_sha,
            num_pages=pdf.page_count,
        )
        pages: list[PageRecord] = []
        blocks: list[BlockRecord] = []
        tables: list[TableRecord] = []
        current_section: str | None = None
        global_block_num = 0
        global_table_num = 0
        previous_text: str | None = None

        page_iter = range(pdf.page_count)
        if self.show_progress and pdf.page_count >= 50:
            page_iter = tqdm(page_iter, desc=f"pages {path.name[:24]}", unit="page", leave=False)

        for page_idx in page_iter:
            page = pdf[page_idx]
            page_num = page_idx + 1
            page_id = stable_hash(doc_id, "page", page_num)
            pages.append(PageRecord(page_id=page_id, doc_id=doc_id, page_num=page_num))

            raw_blocks = self._extract_text_blocks(page)
            if self.merge_short_blocks:
                raw_blocks = self._merge_short_blocks(raw_blocks)

            for raw in raw_blocks:
                text = normalize_ws(raw["text"])
                if len(text) < self.min_chars_per_block:
                    continue
                for chunk_i, chunk in enumerate(chunk_text(text, self.max_block_chars)):
                    block_type = self._classify_block(chunk, raw.get("avg_font", 0.0), raw.get("line_count", 1))
                    if block_type == "section":
                        current_section = chunk[:180]
                    global_block_num += 1
                    block_id = stable_hash(doc_id, page_num, global_block_num, chunk)
                    block = BlockRecord(
                        block_id=block_id,
                        doc_id=doc_id,
                        file_name=path.name,
                        page_num=page_num,
                        block_num=global_block_num,
                        block_type=block_type,
                        section_title=current_section,
                        text=chunk,
                        bbox=raw.get("bbox"),
                        token_estimate=estimate_tokens(chunk),
                        sha1=hashlib.sha1(chunk.encode("utf-8", errors="ignore")).hexdigest(),
                        metadata={
                            "chunk_i": chunk_i,
                            "avg_font": raw.get("avg_font", 0.0),
                            "line_count": raw.get("line_count", 1),
                        },
                    )
                    blocks.append(block)
                    if block_type == "table":
                        global_table_num += 1
                        row_count, col_count = self._estimate_table_shape(chunk)
                        caption = self._infer_table_caption(previous_text, current_section)
                        tables.append(
                            TableRecord(
                                table_id=stable_hash("table", block_id, chunk[:512]),
                                source_block_id=block_id,
                                doc_id=doc_id,
                                file_name=path.name,
                                page_num=page_num,
                                table_num=global_table_num,
                                text=chunk,
                                caption=caption,
                                row_count=row_count,
                                col_count=col_count,
                                sha1=block.sha1,
                                metadata={
                                    "source": "pdf_block_heuristic",
                                    "section_title": current_section,
                                    "bbox": raw.get("bbox"),
                                },
                            )
                        )
                    previous_text = chunk
        pdf.close()
        return doc_record, pages, blocks, tables

    def _extract_text_blocks(self, page: fitz.Page) -> list[dict]:
        info = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
        out: list[dict] = []
        for b in info.get("blocks", []):
            if b.get("type") != 0:
                continue
            lines = b.get("lines", [])
            texts: list[str] = []
            sizes: list[float] = []
            for line in lines:
                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans)
                if line_text.strip():
                    texts.append(line_text.rstrip())
                for s in spans:
                    if s.get("text", "").strip():
                        sizes.append(float(s.get("size", 0.0)))
            text = normalize_ws("\n".join(texts))
            if text:
                out.append(
                    {
                        "text": text,
                        "bbox": [float(x) for x in b.get("bbox", [])],
                        "avg_font": sum(sizes) / len(sizes) if sizes else 0.0,
                        "line_count": len(texts),
                    }
                )
        return out

    def _merge_short_blocks(self, blocks: list[dict]) -> list[dict]:
        merged: list[dict] = []
        carry: dict | None = None
        for b in blocks:
            text = b["text"]
            if carry is None:
                carry = dict(b)
            elif len(carry["text"]) < self.min_chars_per_block * 3 and len(carry["text"]) + len(text) < self.max_block_chars:
                carry["text"] = normalize_ws(carry["text"] + "\n" + text)
                carry["line_count"] = int(carry.get("line_count", 1)) + int(b.get("line_count", 1))
            else:
                merged.append(carry)
                carry = dict(b)
        if carry is not None:
            merged.append(carry)
        return merged

    def _classify_block(self, text: str, avg_font: float, line_count: int) -> str:
        stripped = text.strip()
        numeric_hits = len(_NUMERIC_RE.findall(stripped))
        numeric_ratio = numeric_hits / max(1, len(stripped.split()))
        repeated_spacing = bool(re.search(r"\S\s{2,}\S", stripped))
        pipe_or_tabular = "|" in stripped or "\t" in stripped
        has_table_layout = line_count >= 3 and (repeated_spacing or pipe_or_tabular or numeric_ratio > 0.18)
        if has_table_layout:
            return "table"
        # Headings in filings are usually short, capitalized, and font-prominent.
        if len(stripped) <= 140 and avg_font >= 11.5:
            terminal = stripped[-1:] in {".", ";", ","}
            uppercase_ratio = sum(c.isupper() for c in stripped) / max(1, sum(c.isalpha() for c in stripped))
            if not terminal and (uppercase_ratio > 0.35 or len(stripped.split()) <= 10):
                return "section"
        return "paragraph"

    @staticmethod
    def _estimate_table_shape(text: str) -> tuple[int, int]:
        rows = [r.strip() for r in text.splitlines() if r.strip()]
        if not rows:
            # normalize_ws may have removed line breaks; fall back to a conservative estimate.
            rows = [r.strip() for r in re.split(r"(?<=\))\s+(?=[A-Z])|(?<=\d)\s{2,}", text) if r.strip()]
        col_counts: list[int] = []
        for row in rows[:40]:
            if "\t" in row:
                cells = [c for c in row.split("\t") if c.strip()]
            elif "|" in row:
                cells = [c for c in row.split("|") if c.strip()]
            else:
                cells = [c for c in re.split(r"\s{2,}", row) if c.strip()]
            col_counts.append(len(cells))
        return len(rows), max(col_counts) if col_counts else 0

    @staticmethod
    def _infer_table_caption(previous_text: str | None, current_section: str | None) -> str | None:
        for candidate in (previous_text, current_section):
            if candidate and len(candidate) <= 240 and _TABLE_CAPTION_RE.search(candidate):
                return candidate[:240]
        return current_section[:240] if current_section else None
