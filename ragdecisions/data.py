"""Loading the two datasets and drawing the fixed question sample."""

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATASETS = {
    "multihop_rag": "MultiHop-RAG",
    "natural_questions": "Natural Questions",
}

# HTML tags in Natural Questions pages that end a line of text.
_BLOCK_TAG = re.compile(r"^</?(p|li|h[1-6]|tr|table|ul|ol|dd|dt|dl|br|div)\b", re.IGNORECASE)
_CELL_TAG = re.compile(r"^</(td|th)\b", re.IGNORECASE)
_SPACE_BEFORE = re.compile(r" ([,.;:!?)\]%])")
_SPACE_AFTER = re.compile(r"([(\[$]) ")
_CONTRACTION = re.compile(r" ('s|'re|'ll|'ve|'d|n't)\b")


def render_nq_tokens(tokens, is_html) -> str:
    """Turn a Natural Questions token list into plain text with line breaks.

    Text tokens are joined with spaces, block tags become line breaks, table
    cells are separated by " | ", and spaces before punctuation are removed.
    """
    lines, words = [], []
    for token, tag in zip(tokens, is_html):
        if not tag:
            words.append(token)
        elif _CELL_TAG.match(token):
            words.append("|")
        elif _BLOCK_TAG.match(token) and words:
            lines.append(" ".join(words))
            words = []
    if words:
        lines.append(" ".join(words))
    out = []
    for line in lines:
        line = html.unescape(line).strip(" |")
        line = _SPACE_AFTER.sub(r"\1", _SPACE_BEFORE.sub(r"\1", line))
        line = _CONTRACTION.sub(r"\1", line)
        if line:
            out.append(line)
    return "\n".join(out)


def read_jsonl(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_corpus(dataset: str) -> pd.DataFrame:
    """One row per document: doc_id, title, source, date, text."""
    return pd.DataFrame(read_jsonl(DATA_DIR / dataset / "corpus.jsonl"))


def load_questions(dataset: str) -> pd.DataFrame:
    """One row per question: qid, question, answers, type, gold evidence."""
    return pd.DataFrame(read_jsonl(DATA_DIR / dataset / "questions.jsonl"))


def sample_questions(questions: pd.DataFrame, n: int, strata: str = "type", seed: int = 7) -> pd.DataFrame:
    """Draw n questions, keeping each stratum's share of the full set.

    Shares are rounded with the largest remainder method so the sample has
    exactly n questions, and the draw is fixed by the seed.
    """
    if n > len(questions):
        raise ValueError("sample is larger than the question set")
    counts = questions[strata].value_counts().sort_index()
    exact = counts / counts.sum() * n
    take = np.floor(exact).astype(int)
    leftover = n - take.sum()
    take[(exact - take).sort_values(ascending=False).index[:leftover]] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for stratum, k in take.items():
        rows = questions.index[questions[strata] == stratum].to_numpy()
        picked.extend(rng.choice(rows, size=int(k), replace=False))
    return questions.loc[sorted(picked)].reset_index(drop=True)
