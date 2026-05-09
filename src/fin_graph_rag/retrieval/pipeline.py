from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from fin_graph_rag.config import CrossEncoderConfig, RetrievalConfig, RerankerConfig
from fin_graph_rag.ingest.graph_store import GraphStore
from fin_graph_rag.models import RetrievedBlock, RetrievalCandidate
from fin_graph_rag.rerank.cross_encoder_reranker import CrossEncoderReranker, CrossEncoderRerankerConfig
from fin_graph_rag.rerank.lora_reranker import LoraReranker, LoraRerankerConfig
from fin_graph_rag.retrieval.graph import GraphExpander
from fin_graph_rag.retrieval.hybrid import HybridRetriever
from fin_graph_rag.retrieval.merge import dedupe_keep_best, weighted_merge


DENSE_ONLY_VARIANTS = {"dense", "dense_graph"}
BM25_ONLY_VARIANTS = {"bm25"}
HYBRID_VARIANTS = {
    "hybrid",
    "dense_bm25",
    "hybrid_ce",
    "dense_bm25_ce",
    "hybrid_lora",
    "dense_bm25_lora",
    "graph_hybrid",
    "graph_hybrid_ce",
    "graph_hybrid_lora",
}
GRAPH_VARIANTS = {"dense_graph", "graph_hybrid", "graph_hybrid_ce", "graph_hybrid_lora"}
CE_VARIANTS = {"hybrid_ce", "dense_bm25_ce", "graph_hybrid_ce"}
LORA_VARIANTS = {"hybrid_lora", "dense_bm25_lora", "graph_hybrid_lora"}


DISPLAY_LABELS = {
    "dense": "Dense",
    "hybrid": "Dense + BM25",
    "dense_bm25": "Dense + BM25",
    "hybrid_ce": "Dense + BM25 + CE rerank",
    "dense_bm25_ce": "Dense + BM25 + CE rerank",
    "hybrid_lora": "Dense + BM25 + fine-tuned rerank",
    "dense_bm25_lora": "Dense + BM25 + fine-tuned rerank",
    "dense_graph": "Dense + Graph",
    "graph_hybrid": "Dense + BM25 + Graph",
    "graph_hybrid_ce": "Dense + BM25 + Graph + CE rerank",
    "graph_hybrid_lora": "Dense + BM25 + Graph + fine-tuned rerank",
    "bm25": "BM25",
}


class RetrievalPipeline:
    def __init__(
        self,
        store: GraphStore,
        hybrid: HybridRetriever,
        retrieval_config: RetrievalConfig,
        reranker_config: RerankerConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> None:
        self.store = store
        self.hybrid = hybrid
        self.config = retrieval_config
        self.graph_expander = GraphExpander(store, retrieval_config.graph)
        self.reranker = None
        if reranker_config and reranker_config.enabled:
            self.reranker = LoraReranker(
                LoraRerankerConfig(
                    base_model=reranker_config.base_model,
                    adapter_path=reranker_config.adapter_path,
                    max_length=reranker_config.max_length,
                    batch_size=reranker_config.batch_size,
                    device=reranker_config.device,
                    dtype=reranker_config.dtype,
                )
            )
        self.cross_encoder = None
        if cross_encoder_config and cross_encoder_config.enabled:
            self.cross_encoder = CrossEncoderReranker(
                CrossEncoderRerankerConfig(
                    model_name=cross_encoder_config.model_name,
                    max_length=cross_encoder_config.max_length,
                    batch_size=cross_encoder_config.batch_size,
                    device=cross_encoder_config.device,
                )
            )

    @classmethod
    def load(
        cls,
        db_path: str | Path,
        index_dir: str | Path,
        retrieval_config: RetrievalConfig,
        reranker_config: RerankerConfig | None = None,
        cross_encoder_config: CrossEncoderConfig | None = None,
    ) -> "RetrievalPipeline":
        store = GraphStore(db_path)
        hybrid = HybridRetriever.load(index_dir, rrf_k=retrieval_config.rrf_k)
        return cls(
            store,
            hybrid,
            retrieval_config,
            reranker_config=reranker_config,
            cross_encoder_config=cross_encoder_config,
        )

    def retrieve(
        self,
        query: str,
        variant: str = "graph_hybrid",
        top_k: int | None = None,
        doc_hints: Sequence[str] | None = None,
        doc_scope: str = "off",
    ) -> tuple[list[RetrievedBlock], float]:
        start = time.perf_counter()
        top_k = top_k or self.config.final_k
        variant = self._normalize_variant(variant)
        effective_doc_hints = list(doc_hints or [])
        # Interactive retrieval often includes the company name/ticker in the
        # natural-language query, e.g. "... for 3M?". If document scoping is
        # enabled but the caller did not pass --doc-hint, infer conservative
        # hints from the query so retrieval can stay inside the intended filing.
        if (doc_scope or "off").lower() != "off" and not effective_doc_hints:
            effective_doc_hints = self.store.infer_doc_hints_from_query(query)
        allowed_block_ids, allowed_doc_ids = self._resolve_doc_scope(effective_doc_hints, doc_scope)

        seed_variant = self._seed_variant(variant)
        seeds = self.hybrid.search(
            query,
            bm25_k=self.config.bm25_k,
            dense_k=self.config.dense_k,
            top_n=self.config.seed_k,
            variant=seed_variant,
            allowed_block_ids=allowed_block_ids,
        )

        if variant in {"dense", "bm25", "hybrid", "dense_bm25"}:
            final_candidates = seeds[:top_k]
        elif variant in CE_VARIANTS - GRAPH_VARIANTS:
            final_candidates = self._rerank_with_cross_encoder(query, seeds, top_k)
        elif variant in LORA_VARIANTS - GRAPH_VARIANTS:
            final_candidates = self._rerank_with_lora(query, seeds, top_k)
        elif variant in GRAPH_VARIANTS:
            graph_candidates = self.graph_expander.expand(
                query, seeds, allowed_block_ids=allowed_block_ids, allowed_doc_ids=allowed_doc_ids
            )
            rerank_top_n = max(top_k, int(getattr(self.config, "rerank_top_n", top_k)))
            # First perform a cheap deterministic merge, then send only the best
            # small pool to neural rerankers. This avoids scoring hundreds of
            # graph-expanded candidates with a local LoRA model on CPU.
            pre_candidates = weighted_merge(
                seeds,
                graph_candidates,
                relation_weights=self.config.graph.relation_weights,
                top_n=rerank_top_n,
            )
            candidate_ids = [c.block_id for c in pre_candidates]
            blocks_by_id = self.store.get_blocks(candidate_ids)
            ce_candidates: list[RetrievalCandidate] = []
            lora_candidates: list[RetrievalCandidate] = []
            candidate_blocks = [blocks_by_id[bid] for bid in candidate_ids if bid in blocks_by_id]
            if variant in CE_VARIANTS and self.cross_encoder and self.cross_encoder.available:
                ce_candidates = self.cross_encoder.score(query, candidate_blocks)
            if variant in LORA_VARIANTS and self.reranker and self.reranker.available:
                lora_candidates = self.reranker.score(query, candidate_blocks)
            if ce_candidates or lora_candidates:
                final_candidates = weighted_merge(
                    pre_candidates,
                    [],
                    relation_weights=self.config.graph.relation_weights,
                    top_n=top_k,
                    ce_candidates=ce_candidates,
                    lora_candidates=lora_candidates,
                )
            else:
                final_candidates = pre_candidates[:top_k]
        else:
            raise ValueError(f"Unknown variant: {variant}")

        retrieved = self._hydrate(final_candidates, max_chars=self.config.max_context_chars)
        latency_ms = (time.perf_counter() - start) * 1000
        return retrieved, latency_ms

    def close(self) -> None:
        self.store.close()

    @staticmethod
    def _normalize_variant(variant: str) -> str:
        v = (variant or "graph_hybrid").lower().replace("-", "_")
        aliases = {
            "dense+bm25": "dense_bm25",
            "dense_bm25_rerank": "dense_bm25_lora",
            "fine_tuned": "dense_bm25_lora",
            "lora": "dense_bm25_lora",
            "ce": "dense_bm25_ce",
            "cross_encoder": "dense_bm25_ce",
            "dense_plus_graph": "dense_graph",
        }
        return aliases.get(v, v)

    @staticmethod
    def _seed_variant(variant: str) -> str:
        if variant in DENSE_ONLY_VARIANTS:
            return "dense"
        if variant in BM25_ONLY_VARIANTS:
            return "bm25"
        if variant in HYBRID_VARIANTS:
            return "hybrid"
        raise ValueError(f"Unknown variant: {variant}")

    def _rerank_with_cross_encoder(self, query: str, seeds: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        if not self.cross_encoder or not self.cross_encoder.available:
            raise RuntimeError(
                "This variant needs cross-encoder reranking. Set --ce-model or use the default CE model."
            )
        rerank_top_n = max(top_k, int(getattr(self.config, "rerank_top_n", top_k)))
        pool = seeds[:rerank_top_n]
        blocks_by_id = self.store.get_blocks([c.block_id for c in pool])
        blocks = [blocks_by_id[c.block_id] for c in pool if c.block_id in blocks_by_id]
        return sorted(self.cross_encoder.score(query, blocks), key=lambda c: c.score, reverse=True)[:top_k]

    def _rerank_with_lora(self, query: str, seeds: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        if not self.reranker or not self.reranker.available:
            raise RuntimeError(
                "This variant needs LoRA reranking. Set --lora-base-model and --lora-adapter-path."
            )
        rerank_top_n = max(top_k, int(getattr(self.config, "rerank_top_n", top_k)))
        pool = seeds[:rerank_top_n]
        blocks_by_id = self.store.get_blocks([c.block_id for c in pool])
        blocks = [blocks_by_id[c.block_id] for c in pool if c.block_id in blocks_by_id]
        return sorted(self.reranker.score(query, blocks), key=lambda c: c.score, reverse=True)[:top_k]

    def _resolve_doc_scope(self, doc_hints: Sequence[str], doc_scope: str) -> tuple[set[str] | None, list[str] | None]:
        """Return allowed block/doc IDs for document-scoped retrieval.

        doc_scope values:
        - off:      retrieve across the whole corpus.
        - fallback: restrict to matched document(s), but fall back to full corpus if
                    hints are missing or cannot be matched.
        - strict:   require hints to resolve to ingested documents. This is the
                    recommended FinanceBench setting because each question is
                    evaluated against its evidence filing.
        """
        doc_scope = (doc_scope or "off").lower()
        if doc_scope not in {"off", "fallback", "strict"}:
            raise ValueError("doc_scope must be one of: off, fallback, strict")
        if doc_scope == "off":
            return None, None

        hints = [str(h).strip() for h in doc_hints if str(h).strip()]
        if not hints:
            if doc_scope == "strict":
                raise ValueError(
                    "Document-scoped retrieval was requested, but this row did not provide document hints. "
                    "Use --doc-scope fallback or --doc-scope off if you want corpus-wide fallback."
                )
            return None, None

        doc_ids = self.store.find_doc_ids_by_name_hint(hints)
        if not doc_ids:
            if doc_scope == "strict":
                available = ", ".join(d.file_name for d in self.store.get_all_documents()[:20])
                raise ValueError(
                    "Document-scoped retrieval could not match FinanceBench document hints to ingested PDFs. "
                    f"Hints: {hints}. Make sure data/pdfs filenames contain the FinanceBench doc_name/company/ticker. "
                    f"Available PDF examples: {available}. "
                    "Or rerun with --doc-scope fallback while debugging."
                )
            return None, None

        allowed = self.store.get_block_ids_for_docs(doc_ids)
        if not allowed and doc_scope == "strict":
            raise ValueError(f"Matched document IDs {doc_ids}, but found no blocks in the graph store.")
        return allowed, doc_ids

    def _hydrate(self, candidates: list[RetrievalCandidate], max_chars: int) -> list[RetrievedBlock]:
        blocks = self.store.get_blocks([c.block_id for c in candidates])
        selected: list[RetrievedBlock] = []
        used_chars = 0
        for c in candidates:
            b = blocks.get(c.block_id)
            if not b:
                continue
            if used_chars + len(b.text) > max_chars and selected:
                continue
            selected.append(
                RetrievedBlock(
                    block=b,
                    score=c.score,
                    sources=list(c.extra.get("sources", [c.source])) if c.extra else [c.source],
                    reasons=[c.reason] if c.reason else [],
                )
            )
            used_chars += len(b.text)
        # Reorder selected evidence for readability after relevance selection.
        selected = sorted(selected, key=lambda rb: (rb.block.file_name, rb.block.page_num, rb.block.block_num))
        return selected
