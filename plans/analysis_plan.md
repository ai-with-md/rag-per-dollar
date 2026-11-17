# Analysis plan

Written on November 15, 2026, before any answers were generated. The notebooks follow this plan. Anything added later is marked as added later.

## The question

Which RAG choices actually improve answers per dollar, and do retrieval scores even predict answer quality?

## Data and sample

- **MultiHop-RAG** (ODC-BY): 609 news articles from late 2023 and 2,556 questions. 300 questions drawn with seed 7, keeping each question type's share (comparison, inference, temporal, and questions the articles cannot answer).
- **Natural Questions** (CC BY-SA 3.0): the first two of seven validation files, 2,147 Wikipedia pages and 1,026 questions with a short answer of five tokens or fewer. 200 questions drawn with seed 7.
- The sample is saved in `results/question_sample.csv` and never redrawn.

## The grid

54 configurations, every combination of:

| Choice | Levels |
|---|---|
| Chunk size | 128, 256, 512 tokens (BERT tokenizer, including the two special tokens) |
| Retriever | keyword search (BM25), all-MiniLM-L6-v2, bge-base-en-v1.5 |
| Reranker | none, or bge-reranker-base re-scoring the top 30 |
| Passages given to the model | 3, 5, 10 |

bge-small-en-v1.5 is scored on retrieval only. Everything else is fixed: one prompt, written before the first run and never tuned on results, Gemma 4 E4B through Ollama at temperature 0, and each passage labelled with its title (plus source and date for news).

## Measures

- **Retrieval.** MultiHop-RAG: the share of a question's evidence sentences that were retrieved, where a sentence counts if at least half its characters sit in retrieved chunks, and whether all of them were. Natural Questions: whether a retrieved chunk from the gold page contains a gold answer.
- **Answer correctness.** Right or wrong. A matching rule decides clear cases, and a Gemma 4 12B judge decides the rest. 150 answers are checked by hand to measure how often the rule and the judge agree with a person. Accuracy is measured on questions the documents can answer. MultiHop-RAG's questions that the articles cannot answer (about 12%) get their own measure: the share the model answers anyway instead of declining.
- **Cost per 1,000 answers.** Prompt and answer tokens counted by Ollama, times a published hosted price for the same model, checked on the day and stated with its date, plus the measured search and reranking time priced at a cloud hourly rate. Building the indexes is a one-off cost, reported separately.
- **Floors.** Guessing the most common answer for each question type (on MultiHop-RAG, where answers are lopsided), and the same model answering with no documents.

## Findings to test

1. **Which choice moves accuracy most per dollar?** For each choice, the average paired difference in accuracy between its levels across all other settings, with 2,000 bootstrap resamples of questions, next to the change in cost.
2. **Do retrieval scores predict answer quality?** Per question: accuracy when all evidence was retrieved against when it was not. Per configuration: rank correlation between retrieval score and accuracy across the 54 configurations, with a bootstrap interval. Both repeated for questions the model gets right and wrong with no documents.
3. **Where do returns stop?** Accuracy against prompt size. The plateau is the smallest prompt size within 2 points of the best. Also how often the model answers a question the articles cannot answer, as prompts grow.
4. **What is the cheapest configuration that meets a bar?** For bars of 1, 2 and 5 points below the best configuration, the cheapest configuration with at least an 80% bootstrap chance of meeting the bar, and how the choice moves as the bar rises.
5. **Does it transfer?** Rank correlation of configuration accuracy between the two datasets, and whether finding 4 picks the same configuration on both.

## What this cannot tell

One generator model, two English datasets with short answers, and a fixed prompt. Results may not hold for long-form answers, other domains, much larger models, or tuned prompts. The README will say so next to the findings.

## Amendment, November 17, 2026 (added later, before any answers were generated)

A timing check of 50 answers, not scored, put the full answer run at about 31 hours on an Apple M4. To bring it to about 7.5 hours, answers are generated for a smaller grid and sample. Retrieval is still scored on the full grid and sample, as above.

- **Answer grid: 24 of the 54 configurations.** Chunk sizes 128 and 512, keyword search and bge-base-en-v1.5, with and without the reranker, and 3, 5 and 10 passages. 256-token chunks and all-MiniLM-L6-v2 are scored on retrieval only, like bge-small-en-v1.5. Retrieval had shown 256 tokens sitting between 128 and 512 on both datasets, and MiniLM behind bge-base on both.
- **Answer sample: 150 MultiHop-RAG questions and 100 Natural Questions,** drawn from the question sample with seed 7, keeping each type's share, and saved in `results/answer_sample.csv`.
- Findings 2, 4 and 5 use the 24 answer configurations. The hand check of 150 answers is unchanged.
- **What it costs:** for one configuration against another, the smallest detectable difference in accuracy rises from about 7 to about 9 points on MultiHop-RAG and from about 8 to about 11 on Natural Questions (notebook 01). Findings that average a choice over its other settings pool several such comparisons.
