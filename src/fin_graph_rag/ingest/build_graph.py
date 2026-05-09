from __future__ import annotations

from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from fin_graph_rag.config import ParseConfig
from fin_graph_rag.generation.llm_client import LLMClient
from fin_graph_rag.ingest.claim_extractor import CausalClaimExtractor
from fin_graph_rag.ingest.graph_store import GraphStore
from fin_graph_rag.ingest.neo4j_store import Neo4jGraphWriter
from fin_graph_rag.ingest.pdf_parser import PDFParser

console = Console()


def build_graph(
    pdf_dir: str | Path,
    db_path: str | Path,
    parse_config: ParseConfig | None = None,
    use_llm_claims: bool = False,
    max_claim_blocks: int | None = None,
    reset: bool = True,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
    neo4j_reset: bool = True,
) -> dict[str, int | str]:
    parse_config = parse_config or ParseConfig()
    parser = PDFParser(
        min_chars_per_block=parse_config.min_chars_per_block,
        max_block_chars=parse_config.max_block_chars,
        merge_short_blocks=parse_config.merge_short_blocks,
        show_progress=True,
    )
    console.print(f"[bold]Parsing PDFs from[/bold] {pdf_dir}")
    docs, pages, blocks, tables = parser.parse_dir(pdf_dir)
    console.print(
        f"Parsed {len(docs)} docs, {len(pages)} pages, {len(blocks)} blocks, {len(tables)} explicit tables"
    )

    store = GraphStore(db_path)
    store.init_schema(reset=reset)
    store.upsert_documents(tqdm(docs, desc="write documents", unit="doc"))
    store.upsert_pages(tqdm(pages, desc="write pages", unit="page"))
    store.upsert_blocks(tqdm(blocks, desc="write blocks", unit="block"))
    store.build_structural_edges(show_progress=True)
    store.upsert_tables(tqdm(tables, desc="write tables", unit="table"))

    llm_client = LLMClient.from_env() if use_llm_claims else None
    if use_llm_claims and not (llm_client and llm_client.available):
        console.print("[yellow]--use-llm-claims was set, but OPENAI_API_KEY is missing. Falling back to regex claims.[/yellow]")
    extractor = CausalClaimExtractor(use_llm=use_llm_claims, llm_client=llm_client, show_progress=True)
    console.print("[bold]Extracting causal claims[/bold]" + (" with LLM" if use_llm_claims else " with regex"))
    claims = extractor.extract_many(blocks, max_blocks=max_claim_blocks)
    store.upsert_claims(tqdm(claims, desc="write claims", unit="claim"))

    counts: dict[str, int | str] = {
        "documents": len(docs),
        "pages": len(pages),
        "blocks": len(blocks),
        "tables": len(tables),
        "claims": len(claims),
    }
    store.close()

    if neo4j_uri:
        if not neo4j_user or not neo4j_password:
            raise ValueError("Neo4j sync requires neo4j_user and neo4j_password.")
        console.print(f"[bold]Syncing graph to Neo4j[/bold] {neo4j_uri}")
        writer = Neo4jGraphWriter(neo4j_uri, neo4j_user, neo4j_password, database=neo4j_database)
        try:
            neo_counts = writer.import_from_sqlite(db_path, reset=neo4j_reset)
            counts.update({f"neo4j_{k}": v for k, v in neo_counts.items()})
        finally:
            writer.close()

    console.print(counts)
    return counts


def sync_neo4j_from_sqlite(
    db_path: str | Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str | None = None,
    reset: bool = True,
    batch_size: int = 1000,
) -> dict[str, int]:
    writer = Neo4jGraphWriter(neo4j_uri, neo4j_user, neo4j_password, database=neo4j_database)
    try:
        return writer.import_from_sqlite(db_path, reset=reset, batch_size=batch_size)
    finally:
        writer.close()
