from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from fin_graph_rag.models import BlockRecord, RetrievalCandidate


class DenseBlockIndex:
    def __init__(self, model_name: str, block_ids: list[str], index: faiss.Index, model: SentenceTransformer | None = None) -> None:
        self.model_name = model_name
        self.block_ids = block_ids
        self.index = index
        self._model = model

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @classmethod
    def build(cls, blocks: Sequence[BlockRecord], model_name: str, batch_size: int = 64) -> "DenseBlockIndex":
        model = SentenceTransformer(model_name)
        texts = [b.text for b in blocks]
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        arr = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        return cls(model_name, [b.block_id for b in blocks], index, model=model)

    def search(self, query: str, k: int = 50, allowed_block_ids: set[str] | None = None) -> list[RetrievalCandidate]:
        # FAISS IndexFlatIP cannot apply a document filter inside the index.
        # For FinanceBench document-scoped evaluation, retrieve the full ranking
        # and filter in Python so the dense channel returns the best blocks
        # within the gold document instead of accidentally falling back to
        # other companies' filings.
        fetch_k = len(self.block_ids) if allowed_block_ids is not None else min(len(self.block_ids), k)
        q = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.asarray(q, dtype="float32"), fetch_k)
        out: list[RetrievalCandidate] = []
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
            if idx < 0:
                continue
            bid = self.block_ids[int(idx)]
            if allowed_block_ids is not None and bid not in allowed_block_ids:
                continue
            out.append(RetrievalCandidate(block_id=bid, score=float(score), source="dense", rank=len(out) + 1))
            if len(out) >= k:
                break
        return out

    def save(self, index_dir: str | Path) -> None:
        p = Path(index_dir)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "dense.faiss"))
        pd.DataFrame({"block_id": self.block_ids}).to_parquet(p / "dense_meta.parquet", index=False)
        (p / "dense_model.txt").write_text(self.model_name, encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> "DenseBlockIndex":
        p = Path(index_dir)
        index = faiss.read_index(str(p / "dense.faiss"))
        meta = pd.read_parquet(p / "dense_meta.parquet")
        model_name = (p / "dense_model.txt").read_text(encoding="utf-8").strip()
        return cls(model_name, meta["block_id"].astype(str).tolist(), index)
