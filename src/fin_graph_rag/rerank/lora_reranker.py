from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fin_graph_rag.models import BlockRecord, RetrievalCandidate


@dataclass
class LoraRerankerConfig:
    base_model: str | None
    adapter_path: str | None
    max_length: int = 512
    batch_size: int = 1
    # Default to CPU for local/macOS stability. CUDA can be enabled explicitly
    # with --lora-device cuda. MPS is intentionally opt-in because several
    # transformer/PEFT combinations can segfault on Apple Silicon.
    device: str = "cpu"
    dtype: str = "float32"


class LoraReranker:
    """Inference-only hook for an existing PEFT/LoRA sequence-classifier adapter.

    The project does not train LoRA. It only loads an existing adapter and returns
    per-query candidate relevance scores that can be merged into GraphRAG retrieval.
    """

    def __init__(self, config: LoraRerankerConfig) -> None:
        self.config = config
        self.available = bool(config.base_model and config.adapter_path)
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.torch = None

    def _load(self) -> None:
        if self._loaded or not self.available:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "LoRA reranking requires optional dependencies. Install with: pip install -e '.[lora]'"
            ) from exc

        self.torch = torch

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=1,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, self.config.adapter_path)
        self.model.to(device)
        self.model.eval()
        self._loaded = True

    def _resolve_device(self, torch):
        requested = (self.config.device or "cpu").lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # Keep auto on CPU for macOS stability. Use --lora-device mps only
            # if you explicitly want to test Apple Metal acceleration.
            return torch.device("cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--lora-device cuda was requested, but CUDA is not available.")
        if requested == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError("--lora-device mps was requested, but MPS is not available.")
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self, torch, device):
        requested = (self.config.dtype or "float32").lower()
        if requested in {"fp16", "float16", "half"}:
            # Float16 on CPU is slow and can be unstable for some kernels; keep
            # CPU runs in float32 unless the user explicitly runs CUDA/MPS.
            return torch.float32 if str(device) == "cpu" else torch.float16
        if requested in {"bf16", "bfloat16"}:
            return torch.bfloat16 if str(device) != "cpu" else torch.float32
        return torch.float32

    def score(self, query: str, blocks: Sequence[BlockRecord]) -> list[RetrievalCandidate]:
        if not self.available or not blocks:
            return []
        self._load()
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        candidates: list[RetrievalCandidate] = []
        batch_size = max(1, self.config.batch_size)
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(blocks), batch_size), desc="LoRA rerank", leave=False)
        except Exception:  # pragma: no cover
            iterator = range(0, len(blocks), batch_size)

        with self.torch.no_grad():
            for start in iterator:
                batch = blocks[start : start + batch_size]
                pairs = [f"Question: {query}\n\nCandidate:\n{b.text}" for b in batch]
                encoded = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                device = next(self.model.parameters()).device
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits.squeeze(-1).detach().float().cpu().tolist()
                if isinstance(logits, float):
                    logits = [logits]
                for b, score in zip(batch, logits):
                    candidates.append(
                        RetrievalCandidate(
                            block_id=b.block_id,
                            score=float(score),
                            source="lora",
                            reason="LoRA adapter relevance score",
                        )
                    )
        return candidates
