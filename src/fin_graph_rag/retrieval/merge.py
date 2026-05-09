from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from fin_graph_rag.models import RetrievalCandidate


def minmax_normalize(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    if not candidates:
        return []
    vals = [c.score for c in candidates]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [c.model_copy(update={"score": 1.0}) for c in candidates]
    return [c.model_copy(update={"score": (c.score - lo) / (hi - lo)}) for c in candidates]


def reciprocal_rank_fusion(channels: list[list[RetrievalCandidate]], k: int = 60, top_n: int = 100) -> list[RetrievalCandidate]:
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, list[str]] = defaultdict(list)
    ranks: dict[str, int] = {}
    for channel in channels:
        for rank, c in enumerate(channel, start=1):
            scores[c.block_id] += 1.0 / (k + rank)
            sources[c.block_id].append(c.source)
            ranks[c.block_id] = min(ranks.get(c.block_id, rank), rank)
    out = [
        RetrievalCandidate(
            block_id=bid,
            score=score,
            source="hybrid",
            rank=ranks.get(bid),
            reason="rrf:" + "+".join(sorted(set(sources[bid]))),
            extra={"sources": sorted(set(sources[bid]))},
        )
        for bid, score in scores.items()
    ]
    return sorted(out, key=lambda c: c.score, reverse=True)[:top_n]


def weighted_merge(
    seed_candidates: list[RetrievalCandidate],
    graph_candidates: list[RetrievalCandidate],
    relation_weights: dict[str, float],
    top_n: int,
    lora_candidates: list[RetrievalCandidate] | None = None,
    ce_candidates: list[RetrievalCandidate] | None = None,
) -> list[RetrievalCandidate]:
    seed_norm = minmax_normalize(seed_candidates)
    lora_norm = minmax_normalize(lora_candidates or [])
    ce_norm = minmax_normalize(ce_candidates or [])
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, list[str]] = defaultdict(list)

    for c in seed_norm:
        w = relation_weights.get("seed", 1.0)
        scores[c.block_id] += w * c.score
        sources[c.block_id].add(c.source)
        if c.reason:
            reasons[c.block_id].append(c.reason)

    for c in graph_candidates:
        relation = c.source
        w = relation_weights.get(str(relation), 0.5)
        scores[c.block_id] += w / (1.0 + max(0, c.distance))
        sources[c.block_id].add(str(relation))
        reasons[c.block_id].append(c.reason or str(relation))

    for c in ce_norm:
        scores[c.block_id] += 0.75 * c.score
        sources[c.block_id].add("ce")
        reasons[c.block_id].append("cross-encoder relevance score")

    for c in lora_norm:
        scores[c.block_id] += 0.75 * c.score
        sources[c.block_id].add("lora")
        reasons[c.block_id].append("lora relevance score")

    out = [
        RetrievalCandidate(
            block_id=bid,
            score=score,
            source="graph_hybrid",
            reason="; ".join(reasons[bid][:6]),
            extra={"sources": sorted(sources[bid])},
        )
        for bid, score in scores.items()
    ]
    out = sorted(out, key=lambda c: c.score, reverse=True)[:top_n]
    return [c.model_copy(update={"rank": i + 1}) for i, c in enumerate(out)]


def dedupe_keep_best(candidates: Iterable[RetrievalCandidate]) -> list[RetrievalCandidate]:
    best: dict[str, RetrievalCandidate] = {}
    for c in candidates:
        prev = best.get(c.block_id)
        if prev is None or c.score > prev.score:
            best[c.block_id] = c
    return sorted(best.values(), key=lambda x: x.score, reverse=True)
