"""Small, explicit contracts. Core functionality has no third-party dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math
import re
import tomllib


@dataclass(frozen=True)
class Config:
    provider: str = "openai"
    system: str = "rag"
    model: str = "gpt-5.4-mini-2026-03-17"
    embedding_model: str = "text-embedding-3-small"
    reasoning_effort: str = "none"
    max_output_tokens: int = 1000
    top_k: int = 3
    chunk_tokens: int = 400
    overlap_tokens: int = 60
    tokenizer: str = "cl100k_base"
    timeout_seconds: int = 30
    max_retries: int = 1
    budget_usd: float = 10.0
    input_usd_per_million: float = 0.75
    output_usd_per_million: float = 4.5
    embedding_usd_per_million: float = 0.02
    pricing_checked_at: str = "2026-10-22"
    model_revision: str = ""
    embedding_revision: str = ""
    max_input_tokens: int = 4096
    seed: int = 42

    def __post_init__(self):
        for name in ["top_k", "max_output_tokens", "chunk_tokens", "overlap_tokens",
                     "timeout_seconds", "max_retries"]:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.system not in {"rag", "baseline"}:
            raise ValueError("system must be rag or baseline")
        if self.provider not in {"openai", "huggingface_local"}:
            raise ValueError("Unknown provider")
        if type(self.max_input_tokens) is not int or self.max_input_tokens < 256:
            raise ValueError("max_input_tokens must be an integer >= 256")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.provider == "huggingface_local":
            if any(not re.fullmatch(r"[0-9a-f]{40}", r)
                   for r in [self.model_revision, self.embedding_revision]):
                raise ValueError("Local models require pinned 40-character commit revisions")
            if self.embedding_model != "BAAI/bge-small-en-v1.5" or self.tokenizer != "bge_small_v1_5":
                raise ValueError("Local backend uses BGE small with its matching tokenizer")
        if not 1 <= self.top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("overlap must be smaller than chunk size")
        if self.tokenizer not in {"cl100k_base", "regex_word_v1", "bge_small_v1_5"}:
            raise ValueError("Unknown tokenizer")
        if not 1 <= self.max_output_tokens <= 4000:
            raise ValueError("max_output_tokens must be between 1 and 4000")
        if self.timeout_seconds <= 0 or self.max_retries not in {0, 1}:
            raise ValueError("Use a positive timeout and at most one retry")
        prices = [self.input_usd_per_million, self.output_usd_per_million,
                  self.embedding_usd_per_million]
        if (not math.isfinite(self.budget_usd) or self.budget_usd <= 0
                or any(not math.isfinite(v) or v < 0 for v in prices)):
            raise ValueError("Invalid budget or token prices")

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls(**tomllib.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: str
    heading: str
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    url: str
    fetched_at: str | None
    content_hash: str
    paragraphs: tuple[Paragraph, ...]
    kind: str = "official"

    @classmethod
    def from_dict(cls, row: dict) -> Document:
        if "question" in row or "reference_answer" in row:
            raise ValueError("Evaluation records cannot be indexed as documents")
        if row.get("kind", "official") not in {"official", "synthetic_demo"}:
            raise ValueError("Unrecognized corpus kind")
        paragraphs = tuple(Paragraph(**p) for p in row["paragraphs"])
        if not paragraphs or len({p.paragraph_id for p in paragraphs}) != len(paragraphs):
            raise ValueError("Documents need nonempty, unique paragraphs")
        if any(not p.text.strip() for p in paragraphs):
            raise ValueError("Empty paragraph")
        values = {f.name: row[f.name] for f in fields(cls) if f.name in row}
        values["paragraphs"] = paragraphs
        return cls(**values)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    heading: str
    text: str
    spans: tuple[dict, ...]
    content_hash: str

    @classmethod
    def from_dict(cls, row: dict) -> Chunk:
        return cls(**{**row, "spans": tuple(row["spans"])})


ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["answered", "abstained"]},
        "reason": {"type": ["string", "null"],
                   "enum": [None, "insufficient_evidence", "out_of_scope"]},
        "message": {"type": "string"},
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"text": {"type": "string"}, "evidence_ids": {
                "type": "array", "items": {"type": "string"}}},
            "required": ["text", "evidence_ids"]}},
    },
    "required": ["status", "reason", "message", "claims"],
}


def validate_answer(value: dict, allowed_ids: set[str], grounded: bool) -> dict:
    """Validate syntax and provenance, NOT semantic entailment."""
    if not isinstance(value, dict) or set(value) != {"status", "reason", "message", "claims"}:
        raise ValueError("Unexpected answer fields")
    if value["status"] not in {"answered", "abstained"}:
        raise ValueError("Invalid answer status")
    if not isinstance(value["message"], str) or not isinstance(value["claims"], list):
        raise ValueError("Invalid message or claims")
    if value["status"] == "abstained":
        if value["reason"] not in {"insufficient_evidence", "out_of_scope"}:
            raise ValueError("Abstention needs a reason")
        if value["claims"] or not value["message"].strip():
            raise ValueError("Abstention must explain the limitation without factual claims")
        return value
    if value["reason"] is not None or value["message"] or not value["claims"]:
        raise ValueError("Answered output must contain only cited claims, not uncited prose")
    for claim in value["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"text", "evidence_ids"}:
            raise ValueError("Invalid claim fields")
        if not isinstance(claim["text"], str) or not claim["text"].strip():
            raise ValueError("Empty claim")
        ids = claim["evidence_ids"]
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            raise ValueError("Invalid evidence IDs")
        if len(ids) != len(set(ids)) or set(ids) - allowed_ids:
            raise ValueError("Unknown or duplicate citation")
        if grounded and not ids:
            raise ValueError("Every RAG claim needs a citation")
        if not grounded and ids:
            raise ValueError("Baseline cannot invent citations")
    return value
