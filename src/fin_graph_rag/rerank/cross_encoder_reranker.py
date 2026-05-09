from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fin_graph_rag.models import BlockRecord, RetrievalCandidate


@dataclass
class CrossEncoderRerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    batch_size: int = 8
    device: str = "cpu"


class CrossEncoderReranker:
    """Generic cross-encoder reranker baseline.

    This is the comparison point for the slide's "Dense + BM25 + CE rerank".
    It reranks the candidate blocks returned by vanilla hybrid retrieval. It does
    not use the finance LoRA adapter.
    """

    def __init__(self, config: CrossEncoderRerankerConfig) -> None:
        self.config = config
        self.available = bool(config.model_name)
        self._loaded = False
        self.model = None

    def _load(self) -> None:
        if self._loaded or not self.available:
            return
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "Cross-encoder reranking requires sentence-transformers. Install with: pip install -e ."
            ) from exc

        kwargs = {"max_length": self.config.max_length}
        device = (self.config.device or "cpu").lower()
        if device and device != "auto":
            kwargs["device"] = device
        self.model = CrossEncoder(self.config.model_name, **kwargs)
        self._loaded = True

    def score(self, query: str, blocks: Sequence[BlockRecord]) -> list[RetrievalCandidate]:
        if not self.available or not blocks:
            return []
        self._load()
        assert self.model is not None
        pairs = [(query, b.text) for b in blocks]
        scores = self.model.predict(pairs, batch_size=max(1, self.config.batch_size), show_progress_bar=False)
        # sentence-transformers can return numpy arrays, lists, or scalars.
        try:
            score_list = [float(x) for x in scores.tolist()]
        except AttributeError:
            score_list = [float(x) for x in scores]
        return [
            RetrievalCandidate(
                block_id=b.block_id,
                score=float(s),
                source="ce",
                reason=f"cross-encoder score: {self.config.model_name}",
            )
            for b, s in zip(blocks, score_list)
        ]
