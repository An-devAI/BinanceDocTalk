"""Heading-aware windows with character-level evidence provenance."""

from __future__ import annotations

from itertools import groupby
from functools import lru_cache
import re

from .schemas import Chunk, Document

BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


@lru_cache(maxsize=1)
def bge_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("Install requirements-local-model.txt for BGE tokenization") from exc
    return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", revision=BGE_REVISION,
                                        use_fast=True, trust_remote_code=False)


def token_offsets(text: str, tokenizer: str) -> list[tuple[int, int]]:
    if tokenizer == "regex_word_v1":
        return [m.span() for m in re.finditer(r"\S+", text)]
    if tokenizer == "bge_small_v1_5":
        return [tuple(span) for span in bge_tokenizer()(text, add_special_tokens=False,
                return_offsets_mapping=True, truncation=False)["offset_mapping"] if span[1] > span[0]]
    if tokenizer != "cl100k_base":
        raise ValueError("Unknown tokenizer")
    try:
        import tiktoken
    except ImportError as exc:
        raise ValueError("Install the tokenizer extra for cl100k_base; demo uses regex_word_v1") from exc
    encoding = tiktoken.get_encoding("cl100k_base")
    byte_to_character = []
    for index, char in enumerate(text):
        byte_to_character.extend([index] * len(char.encode("utf-8")))
    offsets, cursor = [], 0
    for token in encoding.encode(text, disallowed_special=()):
        size = len(encoding.decode_single_token_bytes(token))
        offsets.append((byte_to_character[cursor], byte_to_character[cursor + size - 1] + 1))
        cursor += size
    return offsets


def chunk_document(document: Document, size: int = 400, overlap: int = 60,
                   tokenizer: str = "cl100k_base") -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Invalid chunk size or overlap")
    chunks = []
    # Explicit definition/exception units are indivisible and retain exact source spans.
    by_id = {p.paragraph_id: p for p in document.paragraphs}
    positions = {p.paragraph_id: i for i, p in enumerate(document.paragraphs)}
    protected = {pid for group in document.context_groups for pid in group}
    units = []
    for ids in document.context_groups:
        paragraphs = [by_id[pid] for pid in ids]
        heading = " / ".join(dict.fromkeys(p.heading for p in paragraphs))
        units.append((min(positions[pid] for pid in ids), heading, paragraphs, True))
    for heading, group in groupby(document.paragraphs, key=lambda p: p.heading):
        remaining = [p for p in group if p.paragraph_id not in protected]
        if remaining:
            units.append((positions[remaining[0].paragraph_id], heading, remaining, False))
    for _, heading, group, atomic in sorted(units, key=lambda unit: unit[0]):
        paragraphs = list(group)
        pieces, paragraph_offsets, cursor = [], [], 0
        for paragraph in paragraphs:
            pieces.append(paragraph.text)
            paragraph_offsets.append((paragraph.paragraph_id, cursor, cursor + len(paragraph.text)))
            cursor += len(paragraph.text) + 2
        text = "\n\n".join(pieces)
        offsets = token_offsets(text, tokenizer)
        if atomic and len(offsets) > size:
            raise ValueError("Definition/exception context group exceeds chunk token budget")
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            left, right = offsets[start][0], offsets[end - 1][1]
            spans = tuple({"paragraph_id": pid, "start": max(left, pstart) - pstart,
                           "end": min(right, pend) - pstart}
                          for pid, pstart, pend in paragraph_offsets
                          if max(left, pstart) < min(right, pend))
            chunks.append(Chunk(
                chunk_id=f"{document.doc_id}:c{len(chunks)+1:04}", doc_id=document.doc_id,
                title=document.title, url=document.url, heading=heading, text=text[left:right],
                spans=spans, content_hash=document.content_hash,
            ))
            if end == len(offsets):
                break
    return chunks
