from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from fin_graph_rag.ingest.graph_store import GraphStore
from fin_graph_rag.models import BlockRecord, CausalClaim, DocumentRecord, PageRecord, TableRecord

_REL_RE = re.compile(r"[^A-Z0-9_]")


def _chunks(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def _rel_type(edge_type: str) -> str:
    rel = _REL_RE.sub("_", edge_type.upper())
    if not rel or rel[0].isdigit():
        rel = f"REL_{rel}"
    return rel


class Neo4jGraphWriter:
    """Mirror the local SQLite graph into Neo4j for visualization.

    Retrieval still reads from SQLite/FAISS/BM25. Neo4j is intentionally used as
    the human-inspectable graph layer: Browser/Bloom can show Document -> Page ->
    Block, Block -> Table, and Block -> Claim -> Risk/P&L relationships.
    """

    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError("Install Neo4j support with: pip install -e '.[neo4j]'") from exc
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def reset(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def init_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT document_node_id IF NOT EXISTS FOR (n:Document) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT page_node_id IF NOT EXISTS FOR (n:Page) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT block_node_id IF NOT EXISTS FOR (n:Block) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT table_node_id IF NOT EXISTS FOR (n:Table) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT claim_node_id IF NOT EXISTS FOR (n:CausalClaim) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT section_node_id IF NOT EXISTS FOR (n:Section) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT risk_node_id IF NOT EXISTS FOR (n:RiskEntity) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT pnl_node_id IF NOT EXISTS FOR (n:PnLDriver) REQUIRE n.node_id IS UNIQUE",
            "CREATE INDEX block_text IF NOT EXISTS FOR (n:Block) ON (n.text)",
            "CREATE INDEX block_type IF NOT EXISTS FOR (n:Block) ON (n.block_type)",
            "CREATE INDEX table_caption IF NOT EXISTS FOR (n:Table) ON (n.caption)",
            "CREATE INDEX claim_driver IF NOT EXISTS FOR (n:CausalClaim) ON (n.pnl_driver)",
        ]
        with self.driver.session(database=self.database) as session:
            for stmt in statements:
                session.run(stmt)

    def import_from_sqlite(self, db_path: str | Path, reset: bool = False, batch_size: int = 1000) -> dict[str, int]:
        store = GraphStore(db_path)
        try:
            docs = store.get_all_documents()
            pages = store.get_all_pages()
            blocks = store.get_all_blocks()
            tables = store.get_all_tables()
            claims = store.get_all_claims()
            edges = [dict(r) for r in store.get_all_edges()]
            section_ids = sorted({e["src_id"] for e in edges if str(e["src_id"]).startswith("section::")} | {e["dst_id"] for e in edges if str(e["dst_id"]).startswith("section::")})
        finally:
            store.close()

        if reset:
            self.reset()
        self.init_schema()
        self._write_documents(docs, batch_size)
        self._write_pages(pages, batch_size)
        self._write_blocks(blocks, batch_size)
        self._write_tables(tables, batch_size)
        self._write_claims_and_entities(claims, batch_size)
        self._write_sections(section_ids, batch_size)
        self._write_edges(edges, batch_size)
        return {
            "documents": len(docs),
            "pages": len(pages),
            "blocks": len(blocks),
            "tables": len(tables),
            "claims": len(claims),
            "sections": len(section_ids),
            "edges": len(edges),
        }

    def _run_batches(self, label: str, cypher: str, rows: list[dict[str, Any]], batch_size: int) -> None:
        if not rows:
            return
        with self.driver.session(database=self.database) as session:
            for batch in tqdm(list(_chunks(rows, batch_size)), desc=label, unit="batch"):
                session.run(cypher, rows=batch)

    def _write_documents(self, docs: list[DocumentRecord], batch_size: int) -> None:
        rows = [d.model_dump() | {"node_id": d.doc_id} for d in docs]
        self._run_batches(
            "neo4j documents",
            """
            UNWIND $rows AS row
            MERGE (d:Document {node_id: row.node_id})
            SET d.doc_id = row.doc_id,
                d.file_name = row.file_name,
                d.path = row.path,
                d.sha1 = row.sha1,
                d.num_pages = row.num_pages
            """,
            rows,
            batch_size,
        )

    def _write_pages(self, pages: list[PageRecord], batch_size: int) -> None:
        rows = [p.model_dump() | {"node_id": p.page_id} for p in pages]
        self._run_batches(
            "neo4j pages",
            """
            UNWIND $rows AS row
            MERGE (p:Page {node_id: row.node_id})
            SET p.page_id = row.page_id,
                p.doc_id = row.doc_id,
                p.page_num = row.page_num
            """,
            rows,
            batch_size,
        )

    def _write_blocks(self, blocks: list[BlockRecord], batch_size: int) -> None:
        rows = []
        for b in blocks:
            row = b.model_dump()
            row["node_id"] = b.block_id
            row["bbox_json"] = json.dumps(b.bbox)
            row["metadata_json"] = json.dumps(b.metadata)
            rows.append(row)
        self._run_batches(
            "neo4j blocks",
            """
            UNWIND $rows AS row
            MERGE (b:Block {node_id: row.node_id})
            SET b.block_id = row.block_id,
                b.doc_id = row.doc_id,
                b.file_name = row.file_name,
                b.page_num = row.page_num,
                b.block_num = row.block_num,
                b.block_type = row.block_type,
                b.section_title = row.section_title,
                b.text = row.text,
                b.text_preview = left(row.text, 500),
                b.bbox_json = row.bbox_json,
                b.token_estimate = row.token_estimate,
                b.sha1 = row.sha1,
                b.metadata_json = row.metadata_json
            FOREACH (_ IN CASE WHEN row.block_type = 'table' THEN [1] ELSE [] END | SET b:TableBlock)
            FOREACH (_ IN CASE WHEN row.block_type = 'section' THEN [1] ELSE [] END | SET b:SectionBlock)
            FOREACH (_ IN CASE WHEN row.block_type = 'paragraph' THEN [1] ELSE [] END | SET b:ParagraphBlock)
            """,
            rows,
            batch_size,
        )

    def _write_tables(self, tables: list[TableRecord], batch_size: int) -> None:
        rows = []
        for t in tables:
            row = t.model_dump()
            row["node_id"] = t.table_id
            row["text_preview"] = t.text[:700]
            row["metadata_json"] = json.dumps(t.metadata)
            rows.append(row)
        self._run_batches(
            "neo4j tables",
            """
            UNWIND $rows AS row
            MERGE (t:Table {node_id: row.node_id})
            SET t.table_id = row.table_id,
                t.source_block_id = row.source_block_id,
                t.doc_id = row.doc_id,
                t.file_name = row.file_name,
                t.page_num = row.page_num,
                t.table_num = row.table_num,
                t.caption = row.caption,
                t.row_count = row.row_count,
                t.col_count = row.col_count,
                t.sha1 = row.sha1,
                t.text_preview = row.text_preview,
                t.metadata_json = row.metadata_json
            """,
            rows,
            batch_size,
        )


    def _write_sections(self, section_ids: list[str], batch_size: int) -> None:
        rows = []
        for sid in section_ids:
            parts = sid.split("::", 2)
            rows.append({"node_id": sid, "doc_id": parts[1] if len(parts) > 1 else None, "title": parts[2] if len(parts) > 2 else sid})
        self._run_batches(
            "neo4j sections",
            """
            UNWIND $rows AS row
            MERGE (s:Section {node_id: row.node_id})
            SET s.doc_id = row.doc_id,
                s.title = row.title
            """,
            rows,
            batch_size,
        )

    def _write_claims_and_entities(self, claims: list[CausalClaim], batch_size: int) -> None:
        rows = [c.model_dump() | {"node_id": c.claim_id} for c in claims]
        self._run_batches(
            "neo4j claims",
            """
            UNWIND $rows AS row
            MERGE (c:CausalClaim {node_id: row.node_id})
            SET c.claim_id = row.claim_id,
                c.source_block_id = row.source_block_id,
                c.claim_text = row.claim_text,
                c.risk_entity = row.risk_entity,
                c.pnl_driver = row.pnl_driver,
                c.direction = row.direction,
                c.confidence = row.confidence,
                c.extractor = row.extractor
            """,
            rows,
            batch_size,
        )
        risk_rows = sorted({c.risk_entity for c in claims if c.risk_entity})
        pnl_rows = sorted({c.pnl_driver for c in claims if c.pnl_driver})
        self._run_batches(
            "neo4j risk entities",
            """
            UNWIND $rows AS row
            MERGE (r:RiskEntity {node_id: row.node_id})
            SET r.name = row.name
            """,
            [{"node_id": f"risk::{name}", "name": name} for name in risk_rows],
            batch_size,
        )
        self._run_batches(
            "neo4j pnl drivers",
            """
            UNWIND $rows AS row
            MERGE (p:PnLDriver {node_id: row.node_id})
            SET p.name = row.name
            """,
            [{"node_id": f"pnl::{name}", "name": name} for name in pnl_rows],
            batch_size,
        )

    def _write_edges(self, edges: list[dict[str, Any]], batch_size: int) -> None:
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in edges:
            by_type[_rel_type(str(e["edge_type"]))].append(
                {
                    "src_id": e["src_id"],
                    "dst_id": e["dst_id"],
                    "weight": float(e.get("weight") or 1.0),
                    "metadata_json": e.get("metadata_json"),
                }
            )
        for rel_type, rows in tqdm(by_type.items(), desc="neo4j relationship types", unit="type"):
            cypher = f"""
            UNWIND $rows AS row
            MATCH (src {{node_id: row.src_id}})
            MATCH (dst {{node_id: row.dst_id}})
            MERGE (src)-[r:{rel_type}]->(dst)
            SET r.weight = row.weight,
                r.metadata_json = row.metadata_json
            """
            self._run_batches(f"neo4j rel {rel_type}", cypher, rows, batch_size)
