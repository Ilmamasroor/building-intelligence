"""Split one section's text into overlapping chunks on paragraph, sentence, then word boundaries."""

from __future__ import annotations

import re

from building_with_rag.ingestion.mongodb_schema import CHUNK_OVERLAP, CHUNK_SIZE

_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


def _cut(text: str, start: int, size: int) -> int:
    """End index for a chunk starting at `start`, preferring paragraph, sentence, word breaks."""
    hard_end = min(start + size, len(text))
    if hard_end == len(text):
        return hard_end
    window = text[start:hard_end]
    floor = size // 2  # do not cut so early that chunks become tiny
    para = window.rfind("\n\n")
    if para >= floor:
        return start + para
    ends = [m.end() for m in _SENTENCE_END.finditer(window) if m.end() >= floor]
    if ends:
        return start + ends[-1]
    space = window.rfind(" ")
    if space >= floor:
        return start + space
    return hard_end


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = _cut(text, start, size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        if not text[start - 1].isspace():  # begin on a word boundary, not mid-word
            space = text.find(" ", start, end)
            start = space + 1 if space != -1 else end
    return chunks
