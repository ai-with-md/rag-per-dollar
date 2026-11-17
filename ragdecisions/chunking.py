"""Cutting documents into chunks of at most a given number of tokens.

Chunks are character spans over the original text, so any piece of gold
evidence can be checked against what retrieval returned. Sentences are kept
whole where they fit; a sentence longer than the limit is cut between words.
"""

import re
from typing import Callable, List, Tuple

Span = Tuple[int, int]

# A sentence ends at . ! or ? followed by space, or at a line break.
_SENTENCE = re.compile(r"[^\n]*?(?:[.!?](?=\s)|\n|$)")
_WORD = re.compile(r"\S+")


def whitespace_tokens(text: str) -> int:
    """Count tokens as whitespace words. Used in tests; real runs pass a model tokenizer."""
    return len(text.split())


def sentence_spans(text: str) -> List[Span]:
    """Character spans of sentences and lines, with surrounding space trimmed."""
    spans = []
    for m in _SENTENCE.finditer(text):
        start, end = m.start(), m.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def _split_long(text: str, span: Span, max_tokens: int, count: Callable[[str], int]) -> List[Tuple[int, int, int]]:
    """Cut one over-long sentence between words into pieces that fit, with their token counts."""
    pieces, start, end, total = [], None, None, 0
    for m in _WORD.finditer(text, span[0], span[1]):
        n = count(m.group())
        if start is not None and total + n <= max_tokens:
            end, total = m.end(), total + n
        else:
            if start is not None:
                pieces.append((start, end, total))
            start, end, total = m.start(), m.end(), n
    if start is not None:
        pieces.append((start, end, total))
    return pieces


def chunk_spans(text: str, max_tokens: int, count: Callable[[str], int] = whitespace_tokens) -> List[Span]:
    """Pack whole sentences into chunks of at most max_tokens tokens.

    Returns (start, end) character offsets into text. Chunks do not overlap
    and together cover every sentence in order. A single word longer than the
    limit becomes its own chunk, since there is nowhere to cut it.

    Each sentence is counted once and counts are added up, so count must be
    additive over pieces separated by whitespace. That holds for whitespace
    words and for the BERT WordPiece tokenizer the embedding models use.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    units = []
    for start, end in sentence_spans(text):
        n = count(text[start:end])
        if n <= max_tokens:
            units.append((start, end, n))
        else:
            units.extend(_split_long(text, (start, end), max_tokens, count))

    chunks, start, end, total = [], None, None, 0
    for u_start, u_end, n in units:
        if start is not None and total + n <= max_tokens:
            end, total = u_end, total + n
        else:
            if start is not None:
                chunks.append((start, end))
            start, end, total = u_start, u_end, n
    if start is not None:
        chunks.append((start, end))
    return chunks


def evidence_found(evidence: Span, retrieved: List[Span], min_share: float = 0.5) -> bool:
    """True if at least min_share of the evidence characters sit inside retrieved spans.

    retrieved holds the retrieved chunks from the evidence's own document. Chunks
    from one chunking never overlap, so their coverage can simply be added up.

    A piece of evidence cut across two chunks still counts when both halves are
    retrieved, and counts when the larger half is retrieved alone.
    """
    length = evidence[1] - evidence[0]
    if length <= 0:
        return False
    covered = 0
    for start, end in sorted(retrieved):
        covered += max(0, min(end, evidence[1]) - max(start, evidence[0]))
    return covered / length >= min_share
