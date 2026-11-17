"""The fixed grid of RAG choices the study compares. It is small on purpose."""

from itertools import product

CHUNK_SIZES = [128, 256, 512]  # tokens, counted with the BERT tokenizer the embedding models share
RETRIEVERS = {
    "bm25": "Keyword search (BM25)",
    "minilm": "all-MiniLM-L6-v2",
    "bge-base": "bge-base-en-v1.5",
}
RETRIEVAL_ONLY = {"bge-small": "bge-small-en-v1.5"}  # scored on retrieval, no answers generated
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
}
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_POOL = 30  # candidates the reranker re-scores
RERANK = [False, True]
TOP_K = [3, 5, 10]
GENERATOR = "gemma4:e4b-it-qat"
JUDGE = "gemma4:12b-it-qat"


def configurations():
    """All 54 answer configurations as dicts, in a fixed order."""
    return [
        {"chunk": c, "retriever": r, "rerank": rr, "k": k}
        for c, r, rr, k in product(CHUNK_SIZES, RETRIEVERS, RERANK, TOP_K)
    ]


def config_name(chunk: int, retriever: str, rerank: bool, k: int) -> str:
    """Short readable name, for example 'bge-base 256 +rerank k5'."""
    return f"{retriever} {chunk}{' +rerank' if rerank else ''} k{k}"
