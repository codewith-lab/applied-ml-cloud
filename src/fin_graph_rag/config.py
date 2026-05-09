from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    pdf_dir: str = "data/pdfs"
    artifact_dir: str = "artifacts"
    graph_db: str = "artifacts/graph.sqlite"
    index_dir: str = "artifacts/index"
    results_dir: str = "artifacts/results"


class ParseConfig(BaseModel):
    min_chars_per_block: int = 25
    max_block_chars: int = 2500
    merge_short_blocks: bool = True


class ClaimsConfig(BaseModel):
    use_llm: bool = False
    max_blocks: int | None = None
    min_confidence: float = 0.35


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64


class GraphRetrievalConfig(BaseModel):
    page_window: int = 1
    block_window: int = 2
    section_limit: int = 12
    claim_limit: int = 40
    entity_limit: int = 40
    relation_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "seed": 1.0,
            "same_page": 0.45,
            "adjacent_block": 0.55,
            "same_section": 0.50,
            "claim_source": 0.75,
            "same_risk_entity": 0.65,
            "same_pnl_driver": 0.65,
            "query_claim_match": 0.85,
        }
    )


class RetrievalConfig(BaseModel):
    bm25_k: int = 80
    dense_k: int = 80
    seed_k: int = 80
    final_k: int = 8
    rerank_top_n: int = 12
    max_context_chars: int = 12000
    rrf_k: int = 60
    graph: GraphRetrievalConfig = Field(default_factory=GraphRetrievalConfig)


class RerankerConfig(BaseModel):
    enabled: bool = False
    base_model: str | None = None
    adapter_path: str | None = None
    max_length: int = 1024
    batch_size: int = 8
    device: str = "cpu"
    dtype: str = "float32"


class CrossEncoderConfig(BaseModel):
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    batch_size: int = 8
    device: str = "cpu"


class LLMConfig(BaseModel):
    base_url: str | None = None
    answer_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    timeout_seconds: int = 90


class EvaluationConfig(BaseModel):
    dataset_name: str = "PatronusAI/financebench"
    split: str = "train"
    max_questions: int | None = 150
    sample_questions: int | None = None
    sample_seed: int = 42
    doc_scope: str = "strict"


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    parse: ParseConfig = Field(default_factory=ParseConfig)
    claims: ClaimsConfig = Field(default_factory=ClaimsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str | Path | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    cfg = AppConfig.model_validate(data)

    # Env overrides keep CLI usage simple.
    if os.getenv("OPENAI_BASE_URL"):
        cfg.llm.base_url = os.getenv("OPENAI_BASE_URL")
    if os.getenv("ANSWER_MODEL"):
        cfg.llm.answer_model = os.getenv("ANSWER_MODEL", cfg.llm.answer_model)
    if os.getenv("JUDGE_MODEL"):
        cfg.llm.judge_model = os.getenv("JUDGE_MODEL", cfg.llm.judge_model)
    if os.getenv("LORA_BASE_MODEL"):
        cfg.reranker.base_model = os.getenv("LORA_BASE_MODEL")
    if os.getenv("LORA_ADAPTER_PATH"):
        cfg.reranker.adapter_path = os.getenv("LORA_ADAPTER_PATH")
    if os.getenv("LORA_DEVICE"):
        cfg.reranker.device = os.getenv("LORA_DEVICE", cfg.reranker.device)
    if os.getenv("LORA_DTYPE"):
        cfg.reranker.dtype = os.getenv("LORA_DTYPE", cfg.reranker.dtype)
    if os.getenv("LORA_BATCH_SIZE"):
        cfg.reranker.batch_size = int(os.getenv("LORA_BATCH_SIZE", str(cfg.reranker.batch_size)))
    if os.getenv("LORA_MAX_LENGTH"):
        cfg.reranker.max_length = int(os.getenv("LORA_MAX_LENGTH", str(cfg.reranker.max_length)))
    if os.getenv("RERANK_TOP_N"):
        cfg.retrieval.rerank_top_n = int(os.getenv("RERANK_TOP_N", str(cfg.retrieval.rerank_top_n)))
    if os.getenv("CE_MODEL"):
        cfg.cross_encoder.model_name = os.getenv("CE_MODEL", cfg.cross_encoder.model_name)
    if os.getenv("CE_DEVICE"):
        cfg.cross_encoder.device = os.getenv("CE_DEVICE", cfg.cross_encoder.device)
    if os.getenv("CE_BATCH_SIZE"):
        cfg.cross_encoder.batch_size = int(os.getenv("CE_BATCH_SIZE", str(cfg.cross_encoder.batch_size)))
    if os.getenv("CE_MAX_LENGTH"):
        cfg.cross_encoder.max_length = int(os.getenv("CE_MAX_LENGTH", str(cfg.cross_encoder.max_length)))
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
