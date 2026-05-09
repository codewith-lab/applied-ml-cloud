from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm

from fin_graph_rag.models import BlockRecord, CausalClaim, DocumentRecord, PageRecord, TableRecord


class GraphStore:
    """SQLite-backed graph store used by the retrieval pipeline.

    Neo4j support is implemented as a synchronization/export layer so retrieval
    remains easy to run locally while the graph can still be visualized in Neo4j.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def close(self) -> None:
        self.conn.close()

    def init_schema(self, reset: bool = False) -> None:
        cur = self.conn.cursor()
        if reset:
            cur.executescript(
                """
                DROP TABLE IF EXISTS documents;
                DROP TABLE IF EXISTS pages;
                DROP TABLE IF EXISTS blocks;
                DROP TABLE IF EXISTS tables;
                DROP TABLE IF EXISTS claims;
                DROP TABLE IF EXISTS edges;
                DROP INDEX IF EXISTS idx_blocks_doc_page;
                DROP INDEX IF EXISTS idx_blocks_section;
                DROP INDEX IF EXISTS idx_tables_source;
                DROP INDEX IF EXISTS idx_tables_doc_page;
                DROP INDEX IF EXISTS idx_claims_source;
                DROP INDEX IF EXISTS idx_claims_risk;
                DROP INDEX IF EXISTS idx_claims_pnl;
                """
            )
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                path TEXT NOT NULL,
                sha1 TEXT NOT NULL,
                num_pages INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pages (
                page_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_num INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blocks (
                block_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                block_num INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                section_title TEXT,
                text TEXT NOT NULL,
                bbox_json TEXT,
                token_estimate INTEGER,
                sha1 TEXT NOT NULL,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS tables (
                table_id TEXT PRIMARY KEY,
                source_block_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                table_num INTEGER NOT NULL,
                text TEXT NOT NULL,
                caption TEXT,
                row_count INTEGER,
                col_count INTEGER,
                sha1 TEXT NOT NULL,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                source_block_id TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                risk_entity TEXT,
                pnl_driver TEXT,
                direction TEXT,
                confidence REAL,
                extractor TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata_json TEXT,
                PRIMARY KEY(src_id, dst_id, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_blocks_doc_page ON blocks(doc_id, page_num, block_num);
            CREATE INDEX IF NOT EXISTS idx_blocks_section ON blocks(doc_id, section_title);
            CREATE INDEX IF NOT EXISTS idx_tables_source ON tables(source_block_id);
            CREATE INDEX IF NOT EXISTS idx_tables_doc_page ON tables(doc_id, page_num, table_num);
            CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source_block_id);
            CREATE INDEX IF NOT EXISTS idx_claims_risk ON claims(risk_entity);
            CREATE INDEX IF NOT EXISTS idx_claims_pnl ON claims(pnl_driver);
            """
        )
        self.conn.commit()

    def upsert_documents(self, docs: Iterable[DocumentRecord]) -> None:
        docs_list = list(docs)
        self.conn.executemany(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?)",
            [(d.doc_id, d.file_name, d.path, d.sha1, d.num_pages) for d in docs_list],
        )
        self.conn.commit()

    def upsert_pages(self, pages: Iterable[PageRecord]) -> None:
        pages_list = list(pages)
        self.conn.executemany(
            "INSERT OR REPLACE INTO pages VALUES (?,?,?)",
            [(p.page_id, p.doc_id, p.page_num) for p in pages_list],
        )
        self.conn.commit()

    def upsert_blocks(self, blocks: Iterable[BlockRecord]) -> None:
        rows = []
        for b in blocks:
            rows.append(
                (
                    b.block_id,
                    b.doc_id,
                    b.file_name,
                    b.page_num,
                    b.block_num,
                    b.block_type,
                    b.section_title,
                    b.text,
                    json.dumps(b.bbox),
                    b.token_estimate,
                    b.sha1,
                    json.dumps(b.metadata),
                )
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def upsert_tables(self, tables: Iterable[TableRecord]) -> None:
        tables_list = list(tables)
        rows = []
        edge_rows: list[tuple[str, str, str, float, str | None]] = []
        for t in tables_list:
            rows.append(
                (
                    t.table_id,
                    t.source_block_id,
                    t.doc_id,
                    t.file_name,
                    t.page_num,
                    t.table_num,
                    t.text,
                    t.caption,
                    t.row_count,
                    t.col_count,
                    t.sha1,
                    json.dumps(t.metadata),
                )
            )
            edge_rows.append((t.source_block_id, t.table_id, "MENTIONS_TABLE", 1.0, None))
            page_id = self.conn.execute(
                "SELECT page_id FROM pages WHERE doc_id=? AND page_num=?", (t.doc_id, t.page_num)
            ).fetchone()
            if page_id:
                edge_rows.append((page_id["page_id"], t.table_id, "HAS_TABLE", 1.0, None))
        if rows:
            self.conn.executemany("INSERT OR REPLACE INTO tables VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        if edge_rows:
            self.upsert_edges(edge_rows)
        self.conn.commit()

    def upsert_claims(self, claims: Iterable[CausalClaim]) -> None:
        claims_list = list(claims)
        rows = [
            (
                c.claim_id,
                c.source_block_id,
                c.claim_text,
                c.risk_entity,
                c.pnl_driver,
                c.direction,
                c.confidence,
                c.extractor,
            )
            for c in claims_list
        ]
        if rows:
            self.conn.executemany("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?)", rows)
            edge_rows = []
            for c in claims_list:
                edge_rows.append((c.source_block_id, c.claim_id, "SUPPORTS_CLAIM", c.confidence, None))
                if c.risk_entity:
                    edge_rows.append((c.claim_id, f"risk::{c.risk_entity}", "MENTIONS_RISK", 1.0, None))
                if c.pnl_driver:
                    edge_rows.append((c.claim_id, f"pnl::{c.pnl_driver}", "IMPACTS_PNL", 1.0, None))
            self.upsert_edges(edge_rows)
        self.conn.commit()

    def upsert_edges(self, edges: Iterable[tuple[str, str, str, float, str | None]]) -> None:
        edge_list = list(edges)
        if not edge_list:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO edges(src_id,dst_id,edge_type,weight,metadata_json) VALUES (?,?,?,?,?)",
            edge_list,
        )
        self.conn.commit()

    def build_structural_edges(self, show_progress: bool = True) -> None:
        edges: list[tuple[str, str, str, float, str | None]] = []
        pages = self.conn.execute("SELECT page_id, doc_id, page_num FROM pages ORDER BY doc_id, page_num").fetchall()
        page_by_doc_num = {(p["doc_id"], p["page_num"]): p["page_id"] for p in pages}
        iterator = tqdm(pages, desc="document/page edges", unit="page") if show_progress else pages
        for p in iterator:
            edges.append((p["doc_id"], p["page_id"], "HAS_PAGE", 1.0, None))
        blocks = self.conn.execute(
            "SELECT block_id, doc_id, page_num, block_num, section_title FROM blocks ORDER BY doc_id, block_num"
        ).fetchall()
        by_doc: dict[str, list[sqlite3.Row]] = {}
        iterator = tqdm(blocks, desc="page/block edges", unit="block") if show_progress else blocks
        for b in iterator:
            by_doc.setdefault(b["doc_id"], []).append(b)
            page_id = page_by_doc_num.get((b["doc_id"], b["page_num"]))
            if page_id:
                edges.append((page_id, b["block_id"], "HAS_BLOCK", 1.0, None))
        doc_items = by_doc.values()
        iterator = tqdm(list(doc_items), desc="block adjacency/section edges", unit="doc") if show_progress else doc_items
        for doc_blocks in iterator:
            for left, right in zip(doc_blocks, doc_blocks[1:]):
                edges.append((left["block_id"], right["block_id"], "NEXT_BLOCK", 1.0, None))
                edges.append((right["block_id"], left["block_id"], "PREV_BLOCK", 1.0, None))
            by_section: dict[str, list[sqlite3.Row]] = {}
            for b in doc_blocks:
                if b["section_title"]:
                    by_section.setdefault(b["section_title"], []).append(b)
            for section, section_blocks in by_section.items():
                section_id = f"section::{doc_blocks[0]['doc_id']}::{section[:80]}"
                for b in section_blocks:
                    edges.append((section_id, b["block_id"], "HAS_SECTION_BLOCK", 1.0, None))
                    edges.append((b["block_id"], section_id, "IN_SECTION", 1.0, None))
        self.upsert_edges(tqdm(edges, desc="write structural edges", unit="edge") if show_progress else edges)

    def get_all_documents(self) -> list[DocumentRecord]:
        rows = self.conn.execute("SELECT * FROM documents ORDER BY file_name").fetchall()
        return [DocumentRecord(**dict(r)) for r in rows]

    def get_all_pages(self) -> list[PageRecord]:
        rows = self.conn.execute("SELECT * FROM pages ORDER BY doc_id, page_num").fetchall()
        return [PageRecord(**dict(r)) for r in rows]

    def get_all_blocks(self) -> list[BlockRecord]:
        rows = self.conn.execute("SELECT * FROM blocks ORDER BY doc_id, block_num").fetchall()
        return [self._row_to_block(r) for r in rows]

    def get_all_tables(self) -> list[TableRecord]:
        rows = self.conn.execute("SELECT * FROM tables ORDER BY doc_id, table_num").fetchall()
        return [self._row_to_table(r) for r in rows]

    def get_all_claims(self) -> list[CausalClaim]:
        rows = self.conn.execute("SELECT * FROM claims ORDER BY source_block_id, confidence DESC").fetchall()
        return [CausalClaim(**dict(r)) for r in rows]

    def get_all_edges(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM edges ORDER BY edge_type, src_id, dst_id").fetchall()

    def get_blocks(self, block_ids: Sequence[str]) -> dict[str, BlockRecord]:
        if not block_ids:
            return {}
        qmarks = ",".join("?" for _ in block_ids)
        rows = self.conn.execute(f"SELECT * FROM blocks WHERE block_id IN ({qmarks})", list(block_ids)).fetchall()
        return {r["block_id"]: self._row_to_block(r) for r in rows}

    def find_doc_ids_by_name_hint(self, hints: Sequence[str]) -> list[str]:
        """Resolve FinanceBench document hints to ingested document IDs.

        FinanceBench rows may identify the evidence document as a file name,
        stem, company/ticker, filing name, URL fragment, or nested metadata
        value. The matching here is intentionally local and deterministic: it
        normalizes punctuation/case and matches against both ingested file names
        and document IDs.
        """
        if not hints:
            return []

        def norm(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", value.lower())

        normalized_hints = [norm(str(h)) for h in hints if str(h).strip()]
        normalized_hints = [h for h in normalized_hints if h]

        # Common company-name/ticker aliases used by ad-hoc retrieval. This also
        # makes explicit flags such as --doc-hint 3M work when the PDF filename
        # uses the ticker MMM instead of the company name.
        aliases = {
            "3m": ["3m", "mmm"],
            "alphabet": ["alphabet", "google", "goog", "googl"],
            "google": ["google", "goog", "googl", "alphabet"],
            "microsoft": ["microsoft", "msft"],
            "apple": ["apple", "aapl"],
            "amazon": ["amazon", "amzn"],
            "tesla": ["tesla", "tsla"],
            "nvidia": ["nvidia", "nvda"],
            "meta": ["meta", "facebook"],
            "facebook": ["facebook", "meta"],
            "walmart": ["walmart", "wmt"],
            "costco": ["costco", "cost"],
            "target": ["target", "tgt"],
            "boeing": ["boeing", "ba"],
            "disney": ["disney", "dis"],
            "netflix": ["netflix", "nflx"],
            "salesforce": ["salesforce", "crm"],
            "adobe": ["adobe", "adbe"],
            "pepsico": ["pepsico", "pep"],
            "cocacola": ["cocacola", "coke", "ko"],
            "jpmorgan": ["jpmorgan", "jpm"],
            "bankofamerica": ["bankofamerica", "bofa", "bac"],
        }
        expanded_hints: list[str] = []
        seen_hints: set[str] = set()
        for h in normalized_hints:
            for x in [h, *aliases.get(h, [])]:
                if x and x not in seen_hints:
                    expanded_hints.append(x)
                    seen_hints.add(x)
        normalized_hints = expanded_hints
        if not normalized_hints:
            return []

        rows = self.conn.execute("SELECT doc_id, file_name FROM documents ORDER BY file_name").fetchall()
        scored: list[tuple[int, str]] = []
        for r in rows:
            file_name = str(r["file_name"])
            stem = Path(file_name).stem
            candidates = {
                norm(file_name),
                norm(stem),
                norm(str(r["doc_id"])),
                norm(file_name.replace("_", " ").replace("-", " ")),
            }
            # Score matches instead of returning every broad company hit. A
            # structured hint like `3M_2018_10K` should beat the broad hint `3M`
            # when both 3M_2018_10K.pdf and 3M_2022_10K.pdf exist.
            best_score = 0
            for h in normalized_hints:
                if len(h) < 2:
                    continue
                for c in candidates:
                    if not c:
                        continue
                    if h == c:
                        best_score = max(best_score, 1000 + len(h))
                    elif h in c or c in h:
                        best_score = max(best_score, len(h))
            if best_score:
                scored.append((best_score, r["doc_id"]))

        if not scored:
            return []
        max_score = max(score for score, _ in scored)
        out: list[str] = []
        seen: set[str] = set()
        for score, doc_id in scored:
            if score == max_score and doc_id not in seen:
                out.append(doc_id)
                seen.add(doc_id)
        return out


    def infer_doc_hints_from_query(self, query: str) -> list[str]:
        """Infer document-name hints from an ad-hoc user query.

        This is intentionally conservative: it extracts likely company/ticker
        tokens from the query, expands a few common aliases, and lets
        find_doc_ids_by_name_hint do the actual file-name matching. It is used
        only for interactive retrieval. FinanceBench eval still uses explicit
        evidence-document metadata from each row.
        """
        text = str(query or "")
        raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9&.\-]*", text)
        normed: list[str] = []

        def norm(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", value.lower())

        stop = {
            "what", "which", "where", "when", "why", "how", "could", "would", "should",
            "risk", "risks", "pressure", "gross", "margin", "margins", "revenue", "sales",
            "income", "expense", "expenses", "cost", "costs", "cash", "flow", "profit", "loss",
            "for", "the", "and", "or", "of", "in", "on", "to", "from", "with", "about",
            "company", "filing", "annual", "report", "10k", "10q", "fy", "year", "quarter",
        }
        for tok in raw:
            n = norm(tok)
            if len(n) >= 2 and n not in stop:
                normed.append(n)

        # Add adjacent n-grams for company names such as Bank of America.
        raw_norm = [norm(t) for t in raw]
        for size in (2, 3):
            for i in range(0, max(0, len(raw_norm) - size + 1)):
                joined = "".join(raw_norm[i : i + size])
                if len(joined) >= 4 and joined not in stop:
                    normed.append(joined)

        aliases = {
            "3m": ["3m", "mmm"],
            "alphabet": ["alphabet", "google", "goog", "googl"],
            "google": ["google", "goog", "googl", "alphabet"],
            "microsoft": ["microsoft", "msft"],
            "apple": ["apple", "aapl"],
            "amazon": ["amazon", "amzn"],
            "tesla": ["tesla", "tsla"],
            "nvidia": ["nvidia", "nvda"],
            "meta": ["meta", "facebook"],
            "facebook": ["facebook", "meta"],
            "walmart": ["walmart", "wmt"],
            "costco": ["costco", "cost"],
            "target": ["target", "tgt"],
            "boeing": ["boeing", "ba"],
            "disney": ["disney", "dis"],
            "netflix": ["netflix", "nflx"],
            "salesforce": ["salesforce", "crm"],
            "adobe": ["adobe", "adbe"],
            "pepsico": ["pepsico", "pep"],
            "cocacola": ["cocacola", "coke", "ko"],
            "jpmorgan": ["jpmorgan", "jpm"],
            "bankofamerica": ["bankofamerica", "bofa", "bac"],
        }
        expanded: list[str] = []
        for n in normed:
            expanded.append(n)
            expanded.extend(aliases.get(n, []))

        # Stable de-duplication while preserving likely order.
        out: list[str] = []
        seen: set[str] = set()
        for h in expanded:
            if h and h not in seen:
                out.append(h)
                seen.add(h)
        return out

    def get_block_ids_for_docs(self, doc_ids: Sequence[str]) -> set[str]:
        if not doc_ids:
            return set()
        qmarks = ",".join("?" for _ in doc_ids)
        rows = self.conn.execute(f"SELECT block_id FROM blocks WHERE doc_id IN ({qmarks})", list(doc_ids)).fetchall()
        return {r["block_id"] for r in rows}

    def get_doc_debug_names(self, doc_ids: Sequence[str]) -> list[str]:
        if not doc_ids:
            return []
        qmarks = ",".join("?" for _ in doc_ids)
        rows = self.conn.execute(
            f"SELECT file_name FROM documents WHERE doc_id IN ({qmarks}) ORDER BY file_name", list(doc_ids)
        ).fetchall()
        return [r["file_name"] for r in rows]

    def get_neighbor_blocks(self, block_id: str, block_window: int = 2, page_window: int = 1) -> list[tuple[str, str, int]]:
        seed = self.conn.execute("SELECT doc_id, page_num, block_num FROM blocks WHERE block_id=?", (block_id,)).fetchone()
        if not seed:
            return []
        rows = self.conn.execute(
            """
            SELECT block_id, page_num, block_num FROM blocks
            WHERE doc_id=? AND page_num BETWEEN ? AND ?
            ORDER BY page_num, block_num
            """,
            (seed["doc_id"], seed["page_num"] - page_window, seed["page_num"] + page_window),
        ).fetchall()
        out: list[tuple[str, str, int]] = []
        for r in rows:
            if r["block_id"] == block_id:
                continue
            dist = abs(int(r["block_num"]) - int(seed["block_num"]))
            if dist <= block_window:
                out.append((r["block_id"], "adjacent_block", dist))
            elif r["page_num"] == seed["page_num"]:
                out.append((r["block_id"], "same_page", 1))
        return out

    def get_same_section_blocks(self, block_id: str, limit: int = 12) -> list[str]:
        seed = self.conn.execute("SELECT doc_id, section_title, block_num FROM blocks WHERE block_id=?", (block_id,)).fetchone()
        if not seed or not seed["section_title"]:
            return []
        rows = self.conn.execute(
            """
            SELECT block_id FROM blocks
            WHERE doc_id=? AND section_title=? AND block_id<>?
            ORDER BY ABS(block_num - ?), block_num
            LIMIT ?
            """,
            (seed["doc_id"], seed["section_title"], block_id, seed["block_num"], limit),
        ).fetchall()
        return [r["block_id"] for r in rows]

    def get_claim_related_blocks(
        self, block_id: str, limit: int = 40, doc_ids: Sequence[str] | None = None
    ) -> list[tuple[str, str]]:
        claims = self.conn.execute(
            "SELECT risk_entity, pnl_driver FROM claims WHERE source_block_id=?", (block_id,)
        ).fetchall()
        out: list[tuple[str, str]] = []
        seen = {block_id}

        doc_clause = ""
        doc_params: list[str] = []
        if doc_ids:
            qmarks = ",".join("?" for _ in doc_ids)
            doc_clause = f" AND b.doc_id IN ({qmarks})"
            doc_params = list(doc_ids)

        for c in claims:
            if c["risk_entity"]:
                rows = self.conn.execute(
                    f"""
                    SELECT c.source_block_id
                    FROM claims c
                    JOIN blocks b ON b.block_id = c.source_block_id
                    WHERE c.risk_entity=?{doc_clause}
                    ORDER BY c.confidence DESC
                    LIMIT ?
                    """,
                    [c["risk_entity"], *doc_params, limit],
                ).fetchall()
                for r in rows:
                    bid = r["source_block_id"]
                    if bid not in seen:
                        seen.add(bid)
                        out.append((bid, "same_risk_entity"))
            if c["pnl_driver"]:
                rows = self.conn.execute(
                    f"""
                    SELECT c.source_block_id
                    FROM claims c
                    JOIN blocks b ON b.block_id = c.source_block_id
                    WHERE c.pnl_driver=?{doc_clause}
                    ORDER BY c.confidence DESC
                    LIMIT ?
                    """,
                    [c["pnl_driver"], *doc_params, limit],
                ).fetchall()
                for r in rows:
                    bid = r["source_block_id"]
                    if bid not in seen:
                        seen.add(bid)
                        out.append((bid, "same_pnl_driver"))
        return out

    def search_claims(
        self,
        risk_terms: Sequence[str],
        pnl_terms: Sequence[str],
        limit: int = 40,
        doc_ids: Sequence[str] | None = None,
    ) -> list[str]:
        if not risk_terms and not pnl_terms:
            return []
        clauses = []
        params: list[str | int] = []
        for t in risk_terms:
            clauses.append("LOWER(c.risk_entity) LIKE ?")
            params.append(f"%{t.lower()}%")
        for t in pnl_terms:
            clauses.append("LOWER(c.pnl_driver) LIKE ?")
            params.append(f"%{t.lower()}%")

        doc_clause = ""
        if doc_ids:
            qmarks = ",".join("?" for _ in doc_ids)
            doc_clause = f" AND b.doc_id IN ({qmarks})"
            params.extend(list(doc_ids))

        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT c.source_block_id
            FROM claims c
            JOIN blocks b ON b.block_id = c.source_block_id
            WHERE ({' OR '.join(clauses)}){doc_clause}
            ORDER BY c.confidence DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [r["source_block_id"] for r in rows]

    @staticmethod
    def _row_to_block(r: sqlite3.Row) -> BlockRecord:
        return BlockRecord(
            block_id=r["block_id"],
            doc_id=r["doc_id"],
            file_name=r["file_name"],
            page_num=int(r["page_num"]),
            block_num=int(r["block_num"]),
            block_type=r["block_type"],
            section_title=r["section_title"],
            text=r["text"],
            bbox=json.loads(r["bbox_json"]) if r["bbox_json"] else None,
            token_estimate=int(r["token_estimate"] or 0),
            sha1=r["sha1"],
            metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
        )

    @staticmethod
    def _row_to_table(r: sqlite3.Row) -> TableRecord:
        return TableRecord(
            table_id=r["table_id"],
            source_block_id=r["source_block_id"],
            doc_id=r["doc_id"],
            file_name=r["file_name"],
            page_num=int(r["page_num"]),
            table_num=int(r["table_num"]),
            text=r["text"],
            caption=r["caption"],
            row_count=int(r["row_count"] or 0),
            col_count=int(r["col_count"] or 0),
            sha1=r["sha1"],
            metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
        )
