"""Generate answers with the local model for the answer sample and the 24 answer configurations.

Usage: python scripts/run_answers.py multihop_rag   (or natural_questions)

The questions are in results/answer_sample.csv (written by notebook 01) and
the configurations in ragdecisions.grid. Needs Ollama running with the model
in ragdecisions.grid.GENERATOR, and scripts/build_retrieval.py finished.
Took 5.3 hours for MultiHop-RAG and 3.8 for Natural Questions on an
Apple M4. Every answer is cached in cache/answers/<dataset>.jsonl as soon as
it arrives, so the run can be stopped and restarted at any time and reruns
are free. Questions are the outer loop, so a partial run already covers the
whole grid for the questions it reached. Writes results/answers_<dataset>.csv
at the end.
"""

import hashlib
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragdecisions.data import RESULTS_DIR, load_corpus, load_questions  # noqa: E402
from ragdecisions.grid import ANSWER_CHUNK_SIZES, ANSWER_RETRIEVERS, GENERATOR, RERANK, TOP_K  # noqa: E402
from ragdecisions.prompts import build_closed_book_prompt, build_prompt, passage_header  # noqa: E402

CACHE = ROOT / "cache"
OLLAMA = "http://localhost:11434/api/generate"
OPTIONS = {"temperature": 0, "seed": 7, "num_predict": 48, "num_ctx": 8192}


def generate(prompt: str) -> dict:
    body = json.dumps({"model": GENERATOR, "prompt": prompt, "stream": False, "think": False,
                       "options": OPTIONS}).encode()
    started = time.perf_counter()
    request = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=600))
    return {"answer": reply["response"].strip(), "prompt_tokens": reply["prompt_eval_count"],
            "answer_tokens": reply["eval_count"], "seconds": round(time.perf_counter() - started, 3)}


def prompt_key(prompt: str) -> str:
    return hashlib.sha256(f"{GENERATOR}|{json.dumps(OPTIONS, sort_keys=True)}|{prompt}".encode()).hexdigest()[:20]


def main(dataset: str) -> None:
    corpus = load_corpus(dataset).set_index("doc_id")
    sample_ids = pd.read_csv(RESULTS_DIR / "answer_sample.csv")
    questions = load_questions(dataset).set_index("qid")
    questions = questions.loc[sample_ids.loc[sample_ids["dataset"] == dataset, "qid"]]

    chunks, rankings = {}, {}
    for size in ANSWER_CHUNK_SIZES:
        chunks[size] = pd.read_parquet(CACHE / dataset / f"chunks_{size}.parquet")
        for key in ANSWER_RETRIEVERS:
            for reranked in RERANK:
                ranked = pd.read_parquet(CACHE / dataset / f"{'rerank' if reranked else 'top'}_{key}_{size}.parquet")
                rankings[size, key, reranked] = ranked.sort_values(["qid", "rank"]).groupby("qid")["chunk"].apply(list)

    cache_path = CACHE / "answers" / f"{dataset}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            record = json.loads(line)
            cache[record["key"]] = record

    def answer(prompt):
        key = prompt_key(prompt)
        if key not in cache:
            cache[key] = {"key": key, **generate(prompt)}
            with open(cache_path, "a") as f:
                f.write(json.dumps(cache[key]) + "\n")
        return cache[key]

    def passage(size, index):
        doc_id, start, end = chunks[size].iloc[index][["doc_id", "start", "end"]]
        doc = corpus.loc[doc_id]
        return passage_header(doc["title"], doc["source"], doc["date"]), doc["text"][start:end]

    rows, started = [], time.time()
    for n, (qid, q) in enumerate(questions.iterrows(), start=1):
        rows.append({"qid": qid, "chunk": 0, "retriever": "none", "rerank": False, "k": 0,
                     **answer(build_closed_book_prompt(q["question"]))})
        for size in ANSWER_CHUNK_SIZES:
            for key in ANSWER_RETRIEVERS:
                for reranked in RERANK:
                    ranked = rankings[size, key, reranked][qid]
                    for k in TOP_K:  # same ranking, growing k: the model reuses the shared prompt start
                        prompt = build_prompt(q["question"], [passage(size, i) for i in ranked[:k]])
                        rows.append({"qid": qid, "chunk": size, "retriever": key, "rerank": reranked, "k": k,
                                     **answer(prompt)})
        elapsed = time.time() - started
        print(f"{time.strftime('%H:%M:%S')} {n}/{len(questions)} questions, "
              f"{elapsed / n:.0f}s per question so far", flush=True)

    table = pd.DataFrame(rows).drop(columns="key")
    table.insert(0, "dataset", dataset)
    table.to_csv(RESULTS_DIR / f"answers_{dataset}.csv", index=False)
    print(f"wrote {len(table):,} answers to results/answers_{dataset}.csv")


if __name__ == "__main__":
    main(sys.argv[1])
