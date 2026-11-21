"""Grade every distinct answer: the matching rule first, the local judge for the rest.

Usage: python scripts/grade_answers.py multihop_rag natural_questions

Needs Ollama running with the model in ragdecisions.grid.JUDGE, and
scripts/run_answers.py finished. Many configurations give the same answer to a
question, so each distinct (question, answer) pair is graded once. Judge replies
are cached in cache/grades/<dataset>.jsonl. Writes results/grades_<dataset>.csv
with one row per pair: correct (True or False, empty if the judge reply could
not be read) and how it was decided. Takes a few minutes per dataset.
"""

import hashlib
import json
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragdecisions.data import RESULTS_DIR, load_questions  # noqa: E402
from ragdecisions.grading import build_judge_prompt, parse_judge_reply, rule_grade  # noqa: E402
from ragdecisions.grid import JUDGE  # noqa: E402

CACHE = ROOT / "cache" / "grades"
OLLAMA = "http://localhost:11434/api/generate"
OPTIONS = {"temperature": 0, "seed": 7, "num_predict": 4, "num_ctx": 2048}


def judge(prompt: str) -> str:
    body = json.dumps({"model": JUDGE, "prompt": prompt, "stream": False, "think": False,
                       "options": OPTIONS}).encode()
    request = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=300))["response"].strip()


def main(dataset: str) -> None:
    questions = load_questions(dataset).set_index("qid")
    answers = pd.read_csv(RESULTS_DIR / f"answers_{dataset}.csv", keep_default_na=False, na_values=[""])
    pairs = answers[["qid", "answer"]].drop_duplicates().reset_index(drop=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{dataset}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            record = json.loads(line)
            cache[record["key"]] = record["reply"]

    rows = []
    for qid, answer in pairs.itertuples(index=False):
        q = questions.loc[qid]
        decided = rule_grade(answer, q["answers"], answerable=q["type"] != "null")
        if decided is not None:
            rows.append({"qid": qid, "answer": answer, "correct": decided, "method": "rule", "judge_reply": ""})
            continue
        prompt = build_judge_prompt(q["question"], q["answers"], answer)
        key = hashlib.sha256(f"{JUDGE}|{json.dumps(OPTIONS, sort_keys=True)}|{prompt}".encode()).hexdigest()[:20]
        if key not in cache:
            cache[key] = judge(prompt)
            with open(cache_path, "a") as f:
                f.write(json.dumps({"key": key, "reply": cache[key]}) + "\n")
        rows.append({"qid": qid, "answer": answer, "correct": parse_judge_reply(cache[key]), "method": "judge",
                     "judge_reply": cache[key]})

    table = pd.DataFrame(rows)
    table.insert(0, "dataset", dataset)
    table.to_csv(RESULTS_DIR / f"grades_{dataset}.csv", index=False)
    unread = table["correct"].isna().sum()
    print(f"{dataset}: {len(table):,} distinct answers, {(table['method'] == 'judge').sum():,} sent to the judge, "
          f"{unread} judge replies unreadable")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        main(name)
