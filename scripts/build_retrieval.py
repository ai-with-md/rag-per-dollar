"""Chunk both corpora, build the search indexes, and retrieve for the fixed question sample.

Stages, each saved under cache/ and skipped when its file already exists:
  1. chunks at 128, 256 and 512 tokens
  2. embeddings of every chunk with MiniLM, bge-small and bge-base
  3. the top 30 chunks per sampled question for every retriever and chunk size
  4. the reranker's scores for those 30, for the three retrievers in the grid
  5. results/timing.csv: search and reranking time per query, used for the cost
     per answer. Measured one query at a time after a warm-up. Run it with
     nothing else using the machine; to measure again, delete the file.

Then writes results/retrieval.csv: one row per question and configuration,
with the retrieval scores the notebooks use. Run after scripts/download_data.py.
Takes about 6 hours on an Apple M4, most of it on Natural Questions.
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragdecisions.chunking import chunk_spans  # noqa: E402
from ragdecisions.data import DATASETS, RESULTS_DIR, load_corpus, load_questions  # noqa: E402
from ragdecisions.grid import (CHUNK_SIZES, EMBEDDING_MODELS, RERANK_POOL, RERANKER_MODEL,  # noqa: E402
                               RETRIEVERS, TOP_K)
from ragdecisions.metrics import answer_in_texts, evidence_recall, normalize_text  # noqa: E402

CACHE = ROOT / "cache"
DEVICE = "mps"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for bge v1.5 queries
TIMING_QUERIES = 50


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def bert_counter():
    from transformers import AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODELS["bge-base"])
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_chunks(dataset, corpus, count):
    for size in CHUNK_SIZES:
        path = CACHE / dataset / f"chunks_{size}.parquet"
        if path.exists():
            continue
        rows = [(doc_id, s, e) for doc_id, text in zip(corpus["doc_id"], corpus["text"])
                for s, e in chunk_spans(text, size - 2, count)]  # room for the two special tokens
        chunks = pd.DataFrame(rows, columns=["doc_id", "start", "end"])
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks.to_parquet(path)
        log(f"{dataset}: {len(chunks):,} chunks of {size} tokens")


def load_chunks(dataset, size, corpus):
    chunks = pd.read_parquet(CACHE / dataset / f"chunks_{size}.parquet")
    text = dict(zip(corpus["doc_id"], corpus["text"]))
    chunks["text"] = [text[d][s:e] for d, s, e in zip(chunks["doc_id"], chunks["start"], chunks["end"])]
    return chunks


def query_text(model_key, question):
    return BGE_QUERY_PREFIX + question if model_key.startswith("bge") else question


def embed_chunks(dataset, corpus):
    from sentence_transformers import SentenceTransformer
    for key, name in EMBEDDING_MODELS.items():
        model = None
        for size in CHUNK_SIZES:
            path = CACHE / dataset / f"emb_{key}_{size}.npy"
            if path.exists():
                continue
            if model is None:
                model = SentenceTransformer(name, device=DEVICE)
                default_max = model.max_seq_length  # MiniLM: 256, bge: 512. Longer text is cut, as in normal use.
            model.max_seq_length = min(size, default_max)
            texts = load_chunks(dataset, size, corpus)["text"].tolist()
            started = time.time()
            vectors = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            np.save(path, vectors.astype(np.float16))
            log(f"{dataset}: embedded {len(texts):,} chunks of {size} with {key} in {time.time() - started:,.0f}s")


def bm25_tokens(text):
    return normalize_text(text).split()


def retrieve(dataset, corpus, sample):
    """Top RERANK_POOL chunks per question for every retriever (grid plus bge-small) and chunk size."""
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    for size in CHUNK_SIZES:
        chunks = None
        for key in ["bm25", *EMBEDDING_MODELS]:
            path = CACHE / dataset / f"top_{key}_{size}.parquet"
            if path.exists():
                continue
            if chunks is None:
                chunks = load_chunks(dataset, size, corpus)
            questions = sample["question"].tolist()
            if key == "bm25":
                index = BM25Okapi([bm25_tokens(t) for t in chunks["text"]])
                scores = np.vstack([index.get_scores(bm25_tokens(q)) for q in questions])
            else:
                model = SentenceTransformer(EMBEDDING_MODELS[key], device=DEVICE)
                vectors = np.load(CACHE / dataset / f"emb_{key}_{size}.npy").astype(np.float32)
                query_vectors = model.encode([query_text(key, q) for q in questions], batch_size=64,
                                             normalize_embeddings=True, show_progress_bar=False)
                scores = query_vectors @ vectors.T
            top = np.argsort(-scores, axis=1)[:, :RERANK_POOL]
            rows = [(qid, rank + 1, int(idx), float(scores[i, idx]))
                    for i, qid in enumerate(sample["qid"]) for rank, idx in enumerate(top[i])]
            pd.DataFrame(rows, columns=["qid", "rank", "chunk", "score"]).to_parquet(path)
            log(f"{dataset}: retrieved with {key} at {size}")


def rerank(dataset, corpus, sample):
    from sentence_transformers import CrossEncoder
    model = None
    for size in CHUNK_SIZES:
        chunks = None
        for key in RETRIEVERS:
            path = CACHE / dataset / f"rerank_{key}_{size}.parquet"
            if path.exists():
                continue
            if model is None:
                model = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=512)
            if chunks is None:
                chunks = load_chunks(dataset, size, corpus)
            top = pd.read_parquet(CACHE / dataset / f"top_{key}_{size}.parquet")
            question = dict(zip(sample["qid"], sample["question"]))
            rows = []
            for qid, group in top.groupby("qid", sort=False):
                pairs = [(question[qid], chunks["text"].iat[c]) for c in group["chunk"]]
                scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
                order = np.argsort(-scores)
                rows.extend((qid, rank + 1, int(group["chunk"].iat[j]), float(scores[j])) for rank, j in enumerate(order))
            pd.DataFrame(rows, columns=["qid", "rank", "chunk", "score"]).to_parquet(path)
            log(f"{dataset}: reranked {key} at {size}")


def seconds_per_call(step, items, warmup=3):
    """Average seconds per call of step(item), timed one at a time after a few warm-up calls."""
    for item in items[:warmup]:
        step(item)
    seconds = []
    for item in items:
        started = time.perf_counter()
        step(item)
        seconds.append(time.perf_counter() - started)
    return float(np.mean(seconds))


def measure_timing(dataset, corpus, sample):
    """Search and reranking time per query, one query at a time as a live system would run them."""
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer
    questions = sample["question"].tolist()[:TIMING_QUERIES]
    models = {key: SentenceTransformer(name, device=DEVICE) for key, name in EMBEDDING_MODELS.items()}
    reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=512)
    rows = []
    for size in CHUNK_SIZES:
        chunks = load_chunks(dataset, size, corpus)
        index = BM25Okapi([bm25_tokens(t) for t in chunks["text"]])
        searches = {"bm25": lambda q: np.argpartition(-index.get_scores(bm25_tokens(q)), RERANK_POOL)[:RERANK_POOL]}
        for key, model in models.items():
            vectors = np.load(CACHE / dataset / f"emb_{key}_{size}.npy").astype(np.float32)
            searches[key] = lambda q, model=model, vectors=vectors, key=key: np.argpartition(
                -(model.encode([query_text(key, q)], normalize_embeddings=True, show_progress_bar=False) @ vectors.T)[0],
                RERANK_POOL)[:RERANK_POOL]
        for key, search in searches.items():
            rows.append({"dataset": dataset, "step": "search", "retriever": key, "chunk": size,
                         "seconds_per_query": seconds_per_call(search, questions)})
        top = pd.read_parquet(CACHE / dataset / f"top_bge-base_{size}.parquet")
        pools = [[(q, chunks["text"].iat[c]) for c in top.loc[top["qid"] == qid, "chunk"]]
                 for qid, q in zip(sample["qid"][:TIMING_QUERIES], questions)]
        rows.append({"dataset": dataset, "step": "rerank", "retriever": "any", "chunk": size,
                     "seconds_per_query": seconds_per_call(
                         lambda pairs: reranker.predict(pairs, batch_size=32, show_progress_bar=False), pools)})
        log(f"{dataset}: timed search and reranking at {size}")
    return rows


def score(dataset, corpus, sample):
    """Retrieval scores for every question, chunk size, retriever, reranker setting and k."""
    rows = []
    for size in CHUNK_SIZES:
        chunks = load_chunks(dataset, size, corpus)
        for key in ["bm25", *EMBEDDING_MODELS]:
            for reranked in [False, True]:
                path = CACHE / dataset / f"{'rerank' if reranked else 'top'}_{key}_{size}.parquet"
                if not path.exists():
                    continue
                ranked = pd.read_parquet(path)
                lists = ranked.sort_values(["qid", "rank"]).groupby("qid")["chunk"].apply(list)
                for q in sample.itertuples():
                    for k in TOP_K:
                        hits = chunks.iloc[lists[q.qid][:k]]
                        row = {"dataset": dataset, "qid": q.qid, "type": q.type, "chunk": size, "retriever": key,
                               "rerank": reranked, "k": k}
                        if dataset == "multihop_rag":
                            share, everything = evidence_recall(q.gold, list(zip(hits["doc_id"], hits["start"], hits["end"])))
                            row.update(evidence_recall=share, all_evidence=everything)
                        else:
                            gold = q.gold[0]["doc_id"]
                            row.update(answer_hit=answer_in_texts(q.answers, hits.loc[hits["doc_id"] == gold, "text"]))
                        rows.append(row)
    return rows


def main():
    # Apple's math library can print false overflow warnings for float32 matrix products.
    # Rankings were checked against float64 and are identical, so the warnings are silenced.
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    sample_ids = pd.read_csv(RESULTS_DIR / "question_sample.csv")
    count = bert_counter()
    timing_path = RESULTS_DIR / "timing.csv"
    timing, scores = [], []
    for dataset in DATASETS:
        corpus = load_corpus(dataset)
        questions = load_questions(dataset)
        sample = questions[questions["qid"].isin(sample_ids.loc[sample_ids["dataset"] == dataset, "qid"])]
        sample = sample.reset_index(drop=True)
        log(f"{dataset}: {len(corpus):,} documents, {len(sample)} sampled questions")
        build_chunks(dataset, corpus, count)
        embed_chunks(dataset, corpus)
        retrieve(dataset, corpus, sample)
        rerank(dataset, corpus, sample)
        scores += score(dataset, corpus, sample)
        if not timing_path.exists():
            timing += measure_timing(dataset, corpus, sample)
    table = pd.DataFrame(scores)
    table.to_csv(RESULTS_DIR / "retrieval.csv", index=False)
    log(f"wrote {len(table):,} rows to results/retrieval.csv")
    if timing:
        pd.DataFrame(timing).to_csv(timing_path, index=False)
        log("wrote results/timing.csv")


if __name__ == "__main__":
    main()
