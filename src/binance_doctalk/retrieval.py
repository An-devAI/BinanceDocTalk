"""Small-corpus BM25 and dense retrieval, with integrity-checked local indexes."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import re

from .chunking import chunk_document
from .io import digest, utc_now, write_json
from .schemas import Chunk, Config, Document

STOP_WORDS = set("a an the is are was were does do how what why when can could i my to of in on for and or with it this that as be at by from".split())


def terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.casefold()) if w not in STOP_WORDS]


def normalize(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(x) for x in vector):
        raise ValueError("Invalid embedding vector")
    norm = math.sqrt(sum(x*x for x in vector))
    if norm == 0:
        raise ValueError("Zero embedding vector")
    return [x / norm for x in vector]


class Index:
    def __init__(self, chunks: list[Chunk], metadata: dict, vectors: list[list[float]] | None = None):
        if not chunks or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Index needs nonempty, unique chunks")
        self.chunks, self.metadata = chunks, metadata
        self.vectors = [normalize(v) for v in vectors] if vectors is not None else None
        if self.vectors is not None:
            if len(self.vectors) != len(chunks) or len({len(v) for v in self.vectors}) != 1:
                raise ValueError("Embedding shape does not match index")
        self.tokens = [terms(c.title + " " + c.heading + " " + c.text) for c in chunks]
        self.counts = [Counter(t) for t in self.tokens]
        self.average_length = sum(map(len, self.tokens)) / len(chunks)
        self.document_frequency = Counter(t for row in self.tokens for t in set(row))

    def search(self, question: str, k: int = 3,
               query_vector: list[float] | None = None) -> list[dict]:
        if not question.strip() or not 1 <= k <= 10:
            raise ValueError("Use a nonempty question and k between 1 and 10")
        if query_vector is not None:
            if self.vectors is None:
                raise ValueError("No dense embeddings in this index")
            query = normalize(query_vector)
            if len(query) != len(self.vectors[0]):
                raise ValueError("Query embedding dimension mismatch")
            scores = [sum(a*b for a, b in zip(query, vector)) for vector in self.vectors]
        else:
            scores = []
            for token_list, counts in zip(self.tokens, self.counts):
                score = 0.0
                for term in set(terms(question)):
                    frequency = counts[term]
                    df = self.document_frequency[term]
                    idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
                    denominator = frequency + 1.5 * (0.25 + 0.75 * len(token_list) / self.average_length)
                    score += idf * frequency * 2.5 / denominator
                scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        return [{"chunk": self.chunks[i], "score": scores[i]} for i in order[:k]]

    def save(self, path: Path) -> None:
        payload = {"metadata": self.metadata, "chunks": [asdict(c) for c in self.chunks],
                   "vectors": self.vectors}
        write_json(path, {"payload": payload, "integrity_hash": digest(payload)})

    @classmethod
    def load(cls, path: Path, expected_corpus_hash: str | None = None) -> Index:
        data = json.loads(path.read_text())
        payload = data["payload"]
        if digest(payload) != data["integrity_hash"]:
            raise ValueError("Index integrity hash mismatch")
        if expected_corpus_hash and payload["metadata"]["corpus_hash"] != expected_corpus_hash:
            raise ValueError("Index belongs to a different corpus version")
        return cls([Chunk.from_dict(c) for c in payload["chunks"]],
                   payload["metadata"], payload["vectors"])


def build_index(documents: list[Document], config: Config, provider=None,
                reviewed: bool = False) -> Index:
    if not documents:
        raise ValueError("Empty corpus")
    if len({d.kind for d in documents}) != 1:
        raise ValueError("Cannot mix official and synthetic documents")
    chunks = [c for d in documents for c in chunk_document(
        d, config.chunk_tokens, config.overlap_tokens, config.tokenizer)]
    metadata = {
        "format_version": 1, "created_at": utc_now(), "kind": documents[0].kind,
        "corpus_hash": digest([{"doc_id": d.doc_id, "content_hash": d.content_hash}
                               for d in sorted(documents, key=lambda d: d.doc_id)]),
        "document_count": len(documents), "chunk_count": len(chunks),
        "tokenizer": config.tokenizer, "chunk_tokens": config.chunk_tokens,
        "overlap_tokens": config.overlap_tokens, "human_reviewed": reviewed,
        "embedding_model": config.embedding_model if provider else None,
        "embedding_revision": config.embedding_revision if provider else "",
    }
    vectors = None
    if provider:
        vectors = []
        for start in range(0, len(chunks), 32):
            vectors.extend(provider.embed([c.text for c in chunks[start:start+32]]))
    return Index(chunks, metadata, vectors)
