"""Text normalization, answer matching and retrieval scores shared by every notebook."""

import re
import unicodedata

from ragdecisions.chunking import evidence_found

_NOT_WORD = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, and turn every run of punctuation or space into one space.

    "Röntgen," and "roentgen" still differ, but "Hit-points" and "hit points" match.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _NOT_WORD.sub(" ", text.lower()).strip()


def answer_in_texts(answers, texts) -> bool:
    """True if any accepted answer appears, as whole words, in any of the texts."""
    targets = [f" {normalize_text(a)} " for a in answers if normalize_text(a)]
    for text in texts:
        padded = f" {normalize_text(text)} "
        if any(t in padded for t in targets):
            return True
    return False


def evidence_recall(gold, retrieved, min_share: float = 0.5):
    """Share of evidence pieces retrieved, and whether all of them were.

    gold: dicts with doc_id, start and end (character offsets in the document).
    retrieved: (doc_id, start, end) tuples for the retrieved chunks.
    A piece counts as retrieved when at least min_share of its characters sit in
    retrieved chunks of its own document. Questions with no evidence return (nan, nan).
    """
    if not gold:
        return float("nan"), float("nan")
    found = [
        evidence_found((g["start"], g["end"]), [(s, e) for d, s, e in retrieved if d == g["doc_id"]], min_share)
        for g in gold
    ]
    return sum(found) / len(found), all(found)
