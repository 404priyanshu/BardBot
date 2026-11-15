# BardBot

BardBot answers questions about three Shakespeare plays: *Hamlet*, *Macbeth*, and *Romeo and Juliet*. It finds the passages that match your question, then a local language model writes a short answer that cites those passages. By default, everything runs on your own computer and no API key is necessary. An optional API mode lets an OpenAI model write the answers (see [API mode](#api-mode-optional)).

This method is called RAG (retrieval-augmented generation):

1. **Retrieval:** an embedding model turns the question into a vector and finds the 5 most similar passages. Each passage has two vectors: one for its text and one for a short summary in modern English. The search combines both scores.
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
| `--api-model MODEL_ID` | Write the answer with an OpenAI API model (see API mode). |
| `--summary-weight W` | Share of the search score from the chunk summaries, 0 to 1 (default 0.7; 0 = text only). |

The output shows the answer, then the sources. A `*` marks each source that the answer cites. One question takes about 4 seconds on an M5 Pro.

## API mode (optional)

In API mode, an OpenAI model writes the answer. The search still runs on your computer, and the prompt is the same as for the local LLM. Each question costs money and needs an internet connection.

1. Create an API key on the OpenAI dashboard.
2. Create a file named `.env` in the project folder with this line, and paste your key after the `=` sign:

   ```
   OPENAI_API_KEY=
   ```

   `.gitignore` lists `.env`, so git does not commit it. An `OPENAI_API_KEY` that you set in the terminal wins over the file.

3. Copy the exact model ID from the model page on the dashboard. Do not use the display name.
4. Ask a question:

   ```bash
   .venv/bin/python bardbot.py --api-model MODEL_ID "Why does Hamlet fear what happens after death?"
   ```

To compare with the local LLM, run the evaluation on the dev questions only. The answers go to `data/evaluation/answers_api.jsonl`, so the local answers in `answers.jsonl` stay:

```bash
.venv/bin/python evaluate_answers.py --api-model MODEL_ID data/evaluation/retrieval_cases.json data/evaluation/retrieval_cases_holdout.json data/evaluation/retrieval_cases_test.json data/evaluation/unanswerable_dev.json
```

These 4 files hold 43 questions. One question sends about 2,500 input tokens and gets back about 200 output tokens, plus the hidden reasoning tokens of reasoning models. Check the model price before you start.

## Pipeline

The data files are already in `data/`. Run these scripts in this order only to build them again:

| Step | Script | Output |
|---|---|---|
| 1 | `prepare_data.py` | One record per speech: `data/processed/*.jsonl` |
| 2 | `validate_data.py` | Checks the records against the XML |
| 3 | `chunk_data.py` | 462 chunks of 200 words with 40 words of overlap: `data/chunks/chunks.jsonl` |
| 4 | `check_tokens.py` | Checks that each chunk fits the embedding model (512 tokens) |
| 5 | `embed_chunks.py` | Chunk vectors: `data/embeddings/chunk_embeddings.npz` |
| 6 | `summarize_chunks.py --model MODEL_ID` | A modern-English summary of each chunk, written by an OpenAI API model: `data/chunks/summaries.jsonl`. Needs an API key and costs money (gpt-6.1-terra used about 199,000 input and 71,000 output tokens). |
| 7 | `embed_summaries.py` | Summary vectors: `data/embeddings/summary_embeddings.npz` |

Step 6 is the only step that needs the internet. Its output is already in `data/`, so the normal search works offline. If you change the chunks, run steps 6 and 7 again. `summarize_chunks.py` continues from the existing file, so delete `summaries.jsonl` first to summarize all chunks again.

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
| `retrieval_cases_test.json` | 16 | Test questions. Written before the reranker work. Later used as development questions for the summary work. |
| `retrieval_cases_test2.json` | 15 | Test questions for the summary work. Used one time only, for the final measurement. |
| `unanswerable_dev.json`, `unanswerable_test.json` | | Questions that the plays do not answer. The correct reply is "no answer". |

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

### Chunk summaries (used by default)

gpt-6.1-terra wrote a summary of 40–70 modern words for each chunk. The search score is `(1 − W) × text score + W × summary score`. W was chosen on the 39 development questions (original + holdout + test):

| Dev set (39) | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| W = 0 (text only) | 10 | 19 | 20 | 24 | 0.383 |
| W = 0.3 | 13 | 23 | 25 | 31 | 0.491 |
| W = 0.5 | 19 | 26 | 29 | 31 | 0.600 |
| **W = 0.7 (default)** | 19 | 27 | **29** | 31 | 0.617 |
| W = 1 (summary only) | 22 | 26 | 29 | 31 | 0.646 |

W = 0.5 to 1 gave almost the same result. The default is 0.7, because it keeps some text score for questions that quote the exact words of the play.

Final measurement on the unseen test2 set (one run, after the choice of W):

| Test2 set (15) | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| W = 0 (text only) | 2 | 6 | 7 | 7 | 0.298 |
| W = 0.7 | 7 | 11 | **12** | 13 | 0.611 |

The summaries almost double the MRR on both sets.

### Answers

These results use the "Evidence first" prompt in `answer.py` and the old search (W = 0), on the 43 development questions. The same passages went to both models.

| 39 answerable + 4 unanswerable | Phi-4-mini (local) | gpt-6.1-terra (API) |
|---|---|---|
| Reference passage among the 5 | 20 | 20 |
| Answer cites the reference, when it was among the 5 | 11 of 20 | **20 of 20** |
| "No answer" replies to answerable questions | 2 | 4 |
| Unanswerable questions handled correctly | 4 of 4 | 4 of 4 * |
| Average time per question | 3.6 s | 3.9 s |

\* The script counts 3 of 4, but the fourth answer said clearly that the passages do not answer the question.

With the old prompt, Phi-4-mini replied "no answer" to 13 of the 16 test questions. The "Evidence first" prompt fixed this for both models. When the reference passage is missing, the API model usually says what the passages do not show. The local model usually gives an answer from the wrong passages.

## Known limits

- **Retrieval still misses some reference passages:** 10 of 39 dev questions and 3 of 15 test2 questions are not in the top 5.
- **The summaries come from an API model.** To build them again, you need an API key. The summaries can contain small errors or knowledge about the play from outside the passage.
- **Small local model errors.** Phi-4-mini sometimes adds facts that are not in the passages, mixes up characters, or quotes lines longer than 15 words.
- **Small question sets.** 54 retrieval questions in total, so one or two questions change a number a lot.
- **The answer results above use the old search.** Run `evaluate_answers.py` again to measure the answers with the summaries.

## Ideas for next steps

- Measure the answers again with the new search.
- Use a larger local model (for example a 7B model). This needs about 15 GB of memory.
- Write more evaluation questions, especially questions that quote the play.
- Try a stronger embedding model, or smaller chunks.

## Licenses of the models

- `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base`: MIT
- `microsoft/Phi-4-mini-instruct`: MIT
- `data/chunks/summaries.jsonl` was written with the OpenAI API model `gpt-6.1-terra`. OpenAI's terms apply to this output.
