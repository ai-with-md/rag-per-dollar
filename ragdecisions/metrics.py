"""Text normalization and answer matching shared by every notebook."""

import re
import unicodedata

_NOT_WORD = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, and turn every run of punctuation or space into one space.

    "Röntgen," and "roentgen" still differ, but "Hit-points" and "hit points" match.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _NOT_WORD.sub(" ", text.lower()).strip()
