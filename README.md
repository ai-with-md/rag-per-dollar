# RAG per dollar

**Which RAG choices actually improve answers per dollar, and do retrieval scores even predict answer quality?** A measured study of chunk size, embedding model, reranking and the number of retrieved passages, on two public question-answering datasets with known answers.

[![tests](https://github.com/ai-with-md/rag-per-dollar/actions/workflows/tests.yml/badge.svg)](https://github.com/ai-with-md/rag-per-dollar/actions/workflows/tests.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Work in progress.** The study plan, including the findings to test and how, was written before any answers were generated: [plans/analysis_plan.md](plans/analysis_plan.md).

## The study

A fixed grid of 54 configurations, every combination of the levels below. All 54 are scored on retrieval. Answers are generated for 24 of them (128 and 512 tokens, keyword search and bge-base), which kept the local answer run to about 9 hours.

| Choice | Levels |
|---|---|
| Chunk size | 128, 256, 512 tokens |
| Retriever | keyword search (BM25), all-MiniLM-L6-v2, bge-base-en-v1.5 |
| Reranker | none, or bge-reranker-base |
| Passages given to the model | 3, 5, 10 |

Answers come from one model, Gemma 4 E4B, run locally. Configurations are scored on retrieval, answer accuracy and cost per 1,000 answers, with bootstrap intervals.

## Notebooks

| # | Question |
|---|---|
| [01](notebooks/01_data_and_floors.ipynb) | What are we testing, and how far does guessing get? |
| [02](notebooks/02_retrieval.ipynb) | Which configurations find the evidence? |
| [03](notebooks/03_answers.ipynb) | How often are the answers right? |
| [04](notebooks/04_retrieval_vs_answers.ipynb) | Do retrieval scores predict the answers? |
| [05](notebooks/05_cost.ipynb) | What does each choice cost, and which is worth it? |

## Data

- **MultiHop-RAG** (Tang and Yang, 2024; ODC-BY): 609 news articles from late 2023 and 2,556 questions whose evidence is spread over several articles.
- **Natural Questions** (Kwiatkowski et al., 2019; CC BY-SA 3.0): real Google search questions answered from full Wikipedia pages. The first two of the seven validation files are used.

Neither dataset is stored in this repo. `scripts/download_data.py` fetches both from Hugging Face.

## Run it

```bash
git clone https://github.com/ai-with-md/rag-per-dollar
cd rag-per-dollar
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[run,notebooks,dev]" evalsize
python scripts/download_data.py
python scripts/build_retrieval.py   # about 6 hours on an Apple M4
ollama pull gemma4:e4b-it-qat
ollama pull gemma4:12b-it-qat
python scripts/run_answers.py multihop_rag        # 5.3 hours, needs Ollama running
python scripts/run_answers.py natural_questions   # 3.8 hours
python scripts/grade_answers.py multihop_rag natural_questions   # about 11 minutes
jupyter notebook notebooks/
```

Run the tests with `pytest`. They need no downloaded data or models.

## Layout

```
ragdecisions/   small helpers: data loading, chunking, text matching, grading, costs, the fixed grid, chart style
notebooks/      one notebook per question, outputs saved
plans/          the analysis plan, written before any answers
results/        the fixed samples, retrieval scores, timings, generated answers, grades and the hand check
tests/          unit tests, run on every push
docs/figures/   charts used in this README
scripts/        data download, retrieval, answer generation and grading
```
