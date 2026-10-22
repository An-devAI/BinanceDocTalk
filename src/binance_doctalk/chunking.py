"""Heading-aware windows with character-level evidence provenance."""

from __future__ import annotations

from itertools import groupby
import re

from .schemas import Chunk, Document


def token_offsets(text: str, tokenizer: str) -> list[tuple[int, int]]:
    if tokenizer == "regex_word_v1":
        return [m.span() for m in re.finditer(r"\S+", text)]
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
    # Adjacent paragraphs under the same heading remain together.
    for heading, group in groupby(document.paragraphs, key=lambda p: p.heading):
        paragraphs = list(group)
        pieces, paragraph_offsets, cursor = [], [], 0
        for paragraph in paragraphs:
            pieces.append(paragraph.text)
            paragraph_offsets.append((paragraph.paragraph_id, cursor, cursor + len(paragraph.text)))
            cursor += len(paragraph.text) + 2
        text = "\n\n".join(pieces)
        offsets = token_offsets(text, tokenizer)
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
