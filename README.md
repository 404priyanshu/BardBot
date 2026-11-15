# BardBot

BardBot answers questions about three Shakespeare plays: *Hamlet*, *Macbeth*, and *Romeo and Juliet*. It finds the passages that match your question, then a local language model writes a short answer that cites those passages. Everything runs on your own computer. No API key is necessary.

This method is called RAG (retrieval-augmented generation):

1. **Retrieval:** an embedding model turns the question into a vector and finds the 5 most similar passages.
2. **Generation:** a local LLM reads the question and the 5 passages and writes the answer.

## Setup

These steps were tested with Python 3.15 on an Apple Silicon Mac (M5 Pro, 24 GB memory).

1. Create a virtual environment and install the packages:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Download the models into `data/models` (about 7.8 GB):

   ```bash
   .venv/bin/python download_models.py
   ```

   - Add `--reranker` to also download the optional reranker (about 1.1 GB).
   - If the download stops and receives no data, start it again with `HF_HUB_DISABLE_XET=1` before the command. It continues from the partial files.

After this step, all scripts work offline.

## Ask a question

```bash
.venv/bin/python bardbot.py "Why does Hamlet fear what happens after death?"
```

Start the command without a question to ask several questions, one at a time. Type `quit` or press Ctrl+D to stop.

| Option | What it does |
|---|---|
| `--top-k N` | Send N passages to the LLM (default 5). |
| `--show-passages` | Also print the full passages. |
| `--rerank` | Re-score the search results with the reranker (off by default; see Results). |
| `--model NAME` | Use another local LLM in `data/models`. |

The output shows the answer, then the sources. A `*` marks each source that the answer cites. One question takes about 4 seconds on an M5 Pro.

## Pipeline

The data files are already in `data/`. Run these scripts in this order only to build them again:

| Step | Script | Output |
|---|---|---|
| 1 | `prepare_data.py` | One record per speech: `data/processed/*.jsonl` |
| 2 | `validate_data.py` | Checks the records against the XML |
| 3 | `chunk_data.py` | 462 chunks of 200 words with 40 words of overlap: `data/chunks/chunks.jsonl` |
| 4 | `check_tokens.py` | Checks that each chunk fits the embedding model (512 tokens) |
| 5 | `embed_chunks.py` | Chunk vectors: `data/embeddings/chunk_embeddings.npz` |

Each chunk has two text fields:

- `text`: the plain words of the play. The output and the evaluation use it.
- `embed_text`: a heading line (play, act, scene), then each speech with its speaker, for example `MACBETH: Is this a dagger…`. The embedding model uses this field.

Other scripts:

- `search_chunks.py`: search only, without an answer.
- `answer.py`: the prompt and the LLM call.
- `inspect_play.py` and `embed_one_chunk.py`: small demo scripts.

## Evaluation

Three question sets are in `data/evaluation/`. Each question has a reference quote from the scene that answers it.

| File | Questions | Use |
|---|---|---|
| `retrieval_cases.json` | 5 | Original development questions |
| `retrieval_cases_holdout.json` | 18 | More development questions |
| `retrieval_cases_test.json` | 16 | Test questions. Written before the reranker work and used only for the final measurement. |

Commands:

```bash
.venv/bin/python evaluate_retrieval.py data/evaluation/retrieval_cases_test.json
.venv/bin/python evaluate_answers.py
```

`evaluate_answers.py` saves every answer in `data/evaluation/answers.jsonl`. Its numbers only check citations, not if an answer is correct. Read the answers to judge that.

## Results

All question sets are small, so treat all numbers as rough. "Hit@5" means that the passage with the reference quote is among the first 5 results.

### Retrieval

| Version | Original (5), Hit@5 | Holdout (18), Hit@5 |
|---|---|---|
| Plain text | 2 | 4 |
| + speaker names | 3 | 8 |
| + speaker names + heading (final) | 5 | 7 |

Final version on the dev set (original + holdout) and the test set:

| Set | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| Dev (23) | 6 | 11 | 12 | 14 | 0.387 |
| Test (16) | 4 | 8 | 8 | 10 | 0.377 |

The test result matches the dev result, so the design choices did not overfit much. Retrieval finds the right passage in the top 5 only about half of the time.

### Reranker (not used by default)

`BAAI/bge-reranker-base` re-scored the top results. On the dev set, it did not beat plain search, so it is off by default:

| Dev set (23) | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| No reranker | 6 | 11 | 12 | 14 | **0.387** |
| Rerank 20 candidates | 5 | 9 | 13 | 16 | 0.366 |
| Rerank 30 candidates | 5 | 8 | 12 | 16 | 0.341 |
| Rerank 50 candidates | 5 | 7 | 10 | 15 | 0.329 |

### Answers (`microsoft/Phi-4-mini-instruct`, 3.8B)

| Set | Reference passage among the 5 | Answer cites a passage | Answer cites the reference passage | "No answer" replies |
|---|---|---|---|---|
| Original (5) | 5 | 4 | 4 | 1 |
| Holdout (18) | 7 | 9 | 3 | 7 |
| Test (16) | 8 | 2 | 2 | 13 |

## Known limits

- **The LLM refuses too often.** On the test set, it replied "The passages do not answer this question" for 13 of 16 questions. For 5 of these, the reference passage was among the 5 passages.
- **Small model errors.** When the model answers, the answers are mostly reasonable, but sometimes it adds facts that are not in the passages or mixes up characters.
- **Retrieval misses about half of the reference passages** in the top 5.
- **Small question sets.** 39 questions in total, so one or two questions change a number a lot.

## Ideas for next steps

- Use a larger local model (for example a 7B model). This needs about 15 GB of memory.
- Make the prompt less strict about refusals, then measure with new test questions.
- Write more evaluation questions.
- Try smaller chunks or a stronger embedding model to improve retrieval.

## Licenses of the models

- `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base`: MIT
- `microsoft/Phi-4-mini-instruct`: MIT
