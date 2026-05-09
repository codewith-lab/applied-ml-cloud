from fin_graph_rag.models import RetrievalCandidate
from fin_graph_rag.retrieval.merge import reciprocal_rank_fusion, weighted_merge


def test_rrf_merges_channels():
    a = [RetrievalCandidate(block_id="b1", score=1, source="bm25"), RetrievalCandidate(block_id="b2", score=0.5, source="bm25")]
    b = [RetrievalCandidate(block_id="b2", score=1, source="dense"), RetrievalCandidate(block_id="b3", score=0.5, source="dense")]
    out = reciprocal_rank_fusion([a, b], top_n=3)
    assert {c.block_id for c in out} == {"b1", "b2", "b3"}
    assert out[0].block_id == "b2"


def test_weighted_merge_dedupes():
    seeds = [RetrievalCandidate(block_id="b1", score=1, source="hybrid")]
    graph = [RetrievalCandidate(block_id="b1", score=1, source="same_page"), RetrievalCandidate(block_id="b2", score=1, source="same_section")]
    out = weighted_merge(seeds, graph, {"seed": 1, "same_page": 0.5, "same_section": 0.5}, top_n=2)
    assert len(out) == 2
