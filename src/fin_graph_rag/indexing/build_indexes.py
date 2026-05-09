from __future__ import annotations

from pathlib import Path

from rich.console import Console

from fin_graph_rag.indexing.bm25 import BM25BlockIndex
from fin_graph_rag.indexing.dense import DenseBlockIndex
from fin_graph_rag.ingest.graph_store import GraphStore

console = Console()


def build_indexes(
    db_path: str | Path,
    index_dir: str | Path,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> dict[str, int | str]:
    store = GraphStore(db_path)
    blocks = store.get_all_blocks()
    store.close()
    if not blocks:
        raise ValueError(f"No blocks found in graph database: {db_path}")

    p = Path(index_dir)
    p.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Building BM25 index[/bold] for {len(blocks)} blocks")
    bm25 = BM25BlockIndex.build(blocks)
    bm25.save(p / "bm25.pkl")

    console.print(f"[bold]Building dense index[/bold] with {embedding_model}")
    dense = DenseBlockIndex.build(blocks, model_name=embedding_model, batch_size=batch_size)
    dense.save(p)
    return {"blocks": len(blocks), "index_dir": str(p), "embedding_model": embedding_model}
