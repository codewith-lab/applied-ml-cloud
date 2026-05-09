from __future__ import annotations

from pathlib import Path

from fin_graph_rag.indexing.bm25 import BM25BlockIndex
from fin_graph_rag.indexing.dense import DenseBlockIndex
from fin_graph_rag.models import RetrievalCandidate
from fin_graph_rag.retrieval.merge import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, bm25: BM25BlockIndex | None = None, dense: DenseBlockIndex | None = None, rrf_k: int = 60) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    @classmethod
    def load(cls, index_dir: str | Path, rrf_k: int = 60) -> "HybridRetriever":
        p = Path(index_dir)
        return cls(BM25BlockIndex.load(p / "bm25.pkl"), DenseBlockIndex.load(p), rrf_k=rrf_k)

    def search(
        self,
        query: str,
        bm25_k: int = 80,
        dense_k: int = 80,
        top_n: int = 80,
        variant: str = "hybrid",
        allowed_block_ids: set[str] | None = None,
    ) -> list[RetrievalCandidate]:
        variant = variant.lower()
        channels: list[list[RetrievalCandidate]] = []
        if variant in {"bm25", "hybrid", "dense_bm25", "hybrid_ce", "dense_bm25_ce", "hybrid_lora", "dense_bm25_lora", "graph_hybrid", "graph_hybrid_ce", "graph_hybrid_lora"}:
            if not self.bm25:
                raise ValueError("BM25 index is not loaded")
            channels.append(self.bm25.search(query, k=bm25_k, allowed_block_ids=allowed_block_ids))
        if variant in {"dense", "hybrid", "dense_bm25", "hybrid_ce", "dense_bm25_ce", "hybrid_lora", "dense_bm25_lora", "dense_graph", "graph_hybrid", "graph_hybrid_ce", "graph_hybrid_lora"}:
            if not self.dense:
                raise ValueError("Dense index is not loaded")
            channels.append(self.dense.search(query, k=dense_k, allowed_block_ids=allowed_block_ids))
        if not channels:
            raise ValueError(f"Unsupported retrieval variant: {variant}")
        if len(channels) == 1:
            return channels[0][:top_n]
        return reciprocal_rank_fusion(channels, k=self.rrf_k, top_n=top_n)
