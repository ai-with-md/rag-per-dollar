"""The one answer prompt used for every configuration. Written before the first run and not tuned on results.

The question comes before the passages, so the prompts for 3, 5 and 10
passages of the same ranking share their start and the model reads that
part once. The question is repeated after the passages, where small models
look for it.
"""

DECLINE = "Insufficient information."

INSTRUCTION = (
    "Answer the question using only the numbered passages below. "
    "Give the shortest answer that fully answers it: a name, a date, a number, a short phrase, or Yes or No. "
    f"If the passages do not contain the answer, reply exactly: {DECLINE}"
)

CLOSED_BOOK_INSTRUCTION = (
    "Answer the question from what you know. "
    "Give the shortest answer that fully answers it: a name, a date, a number, a short phrase, or Yes or No. "
    f"If you do not know the answer, reply exactly: {DECLINE}"
)


def passage_header(title: str, source: str = "", date: str = "") -> str:
    """Title, plus source and date when the document has them."""
    details = ", ".join(x for x in [source, date] if x and x != "Wikipedia")
    return f"{title} ({details})" if details else title


def build_prompt(question: str, passages) -> str:
    """passages: (header, text) pairs, best first."""
    numbered = "\n\n".join(f"[{i}] {header}\n{text}" for i, (header, text) in enumerate(passages, start=1))
    return (f"{INSTRUCTION}\n\nQuestion: {question}\n\nPassages:\n{numbered}\n\n"
            f"Question: {question}\nShort answer:")


def build_closed_book_prompt(question: str) -> str:
    return f"{CLOSED_BOOK_INSTRUCTION}\n\nQuestion: {question}\nShort answer:"
