"""Download the two public datasets and write them in one shared format.

MultiHop-RAG (Tang and Yang, 2024; ODC-BY): 609 news articles and 2,556
questions with short answers and the exact evidence sentences.
Natural Questions (Kwiatkowski et al., 2019; CC BY-SA 3.0): real Google
search questions answered from full Wikipedia pages. Only the first two of
the seven validation files are used (2,238 questions, about 380 MB).

Output, under data/ (never committed):
    data/<dataset>/corpus.jsonl     doc_id, title, source, date, text
    data/<dataset>/questions.jsonl  qid, question, answers, type, gold
Run with `python scripts/download_data.py`. Files already downloaded are kept.
"""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragdecisions.data import DATA_DIR, render_nq_tokens  # noqa: E402

HF = "https://huggingface.co/datasets"
MHR_FILES = {
    "corpus.json": f"{HF}/yixuantt/MultiHopRAG/resolve/main/corpus.json",
    "MultiHopRAG.json": f"{HF}/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json",
}
NQ_FILES = {
    f"validation-0000{i}-of-00007.parquet":
        f"{HF}/google-research-datasets/natural_questions/resolve/main/default/"
        f"validation-0000{i}-of-00007.parquet"
    for i in range(2)
}
NQ_MAX_ANSWER_TOKENS = 5  # the usual open-domain NQ rule: short answers of five tokens or fewer


def fetch(files: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in files.items():
        path = folder / name
        if path.exists():
            continue
        print(f"downloading {name}")
        urllib.request.urlretrieve(url, path)


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):,} rows to {path.relative_to(ROOT)}")


def prepare_multihop_rag() -> None:
    raw = DATA_DIR / "raw" / "multihop_rag"
    fetch(MHR_FILES, raw)
    articles = json.load(open(raw / "corpus.json"))
    questions = json.load(open(raw / "MultiHopRAG.json"))

    doc_id = {a["url"]: f"mhr-{i:04d}" for i, a in enumerate(articles)}
    body = {a["url"]: a["body"] for a in articles}
    corpus = [{"doc_id": doc_id[a["url"]], "title": a["title"], "source": a["source"],
               "date": a["published_at"][:10], "text": a["body"]} for a in articles]

    rows = []
    for i, q in enumerate(questions):
        gold = []
        for ev in q["evidence_list"]:
            start = body[ev["url"]].find(ev["fact"])
            if start < 0:
                raise ValueError(f"evidence not found in its article for question {i}")
            gold.append({"doc_id": doc_id[ev["url"]], "start": start, "end": start + len(ev["fact"])})
        rows.append({"qid": f"mhr-q{i:04d}", "question": q["query"], "answers": [q["answer"]],
                     "type": q["question_type"].replace("_query", ""), "gold": gold})
    write_jsonl(corpus, DATA_DIR / "multihop_rag" / "corpus.jsonl")
    write_jsonl(rows, DATA_DIR / "multihop_rag" / "questions.jsonl")


def prepare_natural_questions() -> None:
    import pyarrow.parquet as pq

    raw = DATA_DIR / "raw" / "nq"
    fetch(NQ_FILES, raw)
    corpus, rows, seen = [], [], {}
    for name in sorted(NQ_FILES):
        table = pq.read_table(raw / name, columns=["id", "document", "question", "annotations"])
        for ex in table.to_pylist():
            doc, tokens = ex["document"], ex["document"]["tokens"]
            if doc["title"] not in seen:
                seen[doc["title"]] = f"nq-{len(seen):04d}"
                corpus.append({"doc_id": seen[doc["title"]], "title": doc["title"], "source": "Wikipedia",
                               "date": "", "text": render_nq_tokens(tokens["token"], tokens["is_html"])})
            answers = []
            for short in ex["annotations"]["short_answers"]:
                for start, end, text in zip(short["start_token"], short["end_token"], short["text"]):
                    if end - start <= NQ_MAX_ANSWER_TOKENS:
                        clean = render_nq_tokens(text.split(), [False] * len(text.split()))
                        if clean and clean not in answers:
                            answers.append(clean)
            if answers:  # keep questions with at least one short answer, as open-domain NQ does
                rows.append({"qid": f"nq-q{ex['id']}", "question": ex["question"]["text"], "answers": answers,
                             "type": "factoid", "gold": [{"doc_id": seen[doc["title"]]}]})
    write_jsonl(corpus, DATA_DIR / "natural_questions" / "corpus.jsonl")
    write_jsonl(rows, DATA_DIR / "natural_questions" / "questions.jsonl")


if __name__ == "__main__":
    prepare_multihop_rag()
    prepare_natural_questions()
