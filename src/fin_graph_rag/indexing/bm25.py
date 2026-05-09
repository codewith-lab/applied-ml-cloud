from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from fin_graph_rag.models import BlockRecord, RetrievalCandidate
from fin_graph_rag.utils.text import tokenize


class BM25BlockIndex:
    def __init__(self, block_ids: list[str], tokenized_corpus: list[list[str]], bm25: BM25Okapi) -> None:
        self.block_ids = block_ids
        self.tokenized_corpus = tokenized_corpus
        self.bm25 = bm25

    @classmethod
    def build(cls, blocks: Sequence[BlockRecord]) -> "BM25BlockIndex":
        block_ids = [b.block_id for b in blocks]
        tokenized = [tokenize(b.text) for b in tqdm(blocks, desc="tokenize BM25", unit="block")]
        return cls(block_ids, tokenized, BM25Okapi(tokenized))

    def search(self, query: str, k: int = 50, allowed_block_ids: set[str] | None = None) -> list[RetrievalCandidate]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        out: list[RetrievalCandidate] = []
        for idx, score in ranked:
            bid = self.block_ids[idx]
            if allowed_block_ids is not None and bid not in allowed_block_ids:
                continue
            if score <= 0:
                continue
            out.append(RetrievalCandidate(block_id=bid, score=float(score), source="bm25", rank=len(out) + 1))
            if len(out) >= k:
                break
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump({"block_ids": self.block_ids, "tokenized_corpus": self.tokenized_corpus, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25BlockIndex":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        return cls(data["block_ids"], data["tokenized_corpus"], data["bm25"])
