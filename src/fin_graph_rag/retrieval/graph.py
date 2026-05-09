from __future__ import annotations

from typing import Sequence

from fin_graph_rag.config import GraphRetrievalConfig
from fin_graph_rag.ingest.claim_extractor import extract_query_terms
from fin_graph_rag.ingest.graph_store import GraphStore
from fin_graph_rag.models import RetrievalCandidate
from fin_graph_rag.retrieval.merge import dedupe_keep_best


class GraphExpander:
    def __init__(self, store: GraphStore, config: GraphRetrievalConfig) -> None:
        self.store = store
        self.config = config

    def expand(
        self,
        query: str,
        seeds: list[RetrievalCandidate],
        allowed_block_ids: set[str] | None = None,
        allowed_doc_ids: Sequence[str] | None = None,
    ) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        seen_seed_ids = {s.block_id for s in seeds}

        def allowed(block_id: str) -> bool:
            return allowed_block_ids is None or block_id in allowed_block_ids

        # Seed-centered expansion: filing structure and claim overlay.
        for seed_rank, seed in enumerate(seeds, start=1):
            for bid, relation, dist in self.store.get_neighbor_blocks(
                seed.block_id,
                block_window=self.config.block_window,
                page_window=self.config.page_window,
            ):
                if bid not in seen_seed_ids and allowed(bid):
                    candidates.append(
                        RetrievalCandidate(
                            block_id=bid,
                            score=1.0 / seed_rank,
                            source=relation,
                            distance=max(1, dist),
                            reason=f"{relation} expansion from seed rank {seed_rank}",
                        )
                    )

            for bid in self.store.get_same_section_blocks(seed.block_id, limit=self.config.section_limit):
                if bid not in seen_seed_ids and allowed(bid):
                    candidates.append(
                        RetrievalCandidate(
                            block_id=bid,
                            score=1.0 / seed_rank,
                            source="same_section",
                            distance=1,
                            reason=f"same section as seed rank {seed_rank}",
                        )
                    )

            for bid, relation in self.store.get_claim_related_blocks(
                seed.block_id, limit=self.config.entity_limit, doc_ids=allowed_doc_ids
            ):
                if bid not in seen_seed_ids and allowed(bid):
                    candidates.append(
                        RetrievalCandidate(
                            block_id=bid,
                            score=1.0 / seed_rank,
                            source=relation,
                            distance=1,
                            reason=f"semantic overlay: {relation} from seed rank {seed_rank}",
                        )
                    )

        # Query-centered expansion: direct match against risk/PnL overlay.
        terms = extract_query_terms(query)
        for bid in self.store.search_claims(
            terms["risk_terms"], terms["pnl_terms"], limit=self.config.claim_limit, doc_ids=allowed_doc_ids
        ):
            if not allowed(bid):
                continue
            candidates.append(
                RetrievalCandidate(
                    block_id=bid,
                    score=1.0,
                    source="query_claim_match",
                    distance=0,
                    reason="query risk/P&L terms matched semantic claim overlay",
                    extra=terms,
                )
            )

        return dedupe_keep_best(candidates)
