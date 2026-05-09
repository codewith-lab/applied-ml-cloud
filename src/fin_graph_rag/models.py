from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

BlockType = Literal["section", "paragraph", "table", "other"]
CandidateSource = Literal[
    "dense",
    "bm25",
    "hybrid",
    "same_page",
    "adjacent_block",
    "same_section",
    "claim_source",
    "same_risk_entity",
    "same_pnl_driver",
    "query_claim_match",
    "lora",
]


class DocumentRecord(BaseModel):
    doc_id: str
    file_name: str
    path: str
    sha1: str
    num_pages: int


class PageRecord(BaseModel):
    page_id: str
    doc_id: str
    page_num: int


class BlockRecord(BaseModel):
    block_id: str
    doc_id: str
    file_name: str
    page_num: int
    block_num: int
    block_type: BlockType
    section_title: Optional[str] = None
    text: str
    bbox: Optional[list[float]] = None
    token_estimate: int = 0
    sha1: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableRecord(BaseModel):
    """Explicit table node derived from a parsed table-like block.

    The source block remains the canonical retrieval unit. The table node is an
    overlay for graph visualization and graph expansion:

        (:Block {block_type:'table'})-[:MENTIONS_TABLE]->(:Table)
    """

    table_id: str
    source_block_id: str
    doc_id: str
    file_name: str
    page_num: int
    table_num: int
    text: str
    caption: Optional[str] = None
    row_count: int = 0
    col_count: int = 0
    sha1: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CausalClaim(BaseModel):
    claim_id: str
    source_block_id: str
    claim_text: str
    risk_entity: Optional[str] = None
    pnl_driver: Optional[str] = None
    direction: Optional[str] = None
    confidence: float = 0.0
    extractor: str = "regex"


class RetrievalCandidate(BaseModel):
    block_id: str
    score: float
    source: CandidateSource | str
    rank: Optional[int] = None
    reason: Optional[str] = None
    distance: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievedBlock(BaseModel):
    block: BlockRecord
    score: float
    sources: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class AnswerResult(BaseModel):
    question: str
    answer: str
    retrieved: list[RetrievedBlock]
    latency_ms: float
    model: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
