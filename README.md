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
| `--top-k N` | Send N passages to the LLM (default 8 in API mode, 5 otherwise). |
| `--rewrite` / `--no-rewrite` | Query rewriting before the search (default: on in API mode; needs API mode). |
| `--show-passages` | Also print the full passages. |
| `--rerank` | Re-score the search results with the reranker (off by default; see Results). |
| `--model NAME` | Use another local LLM in `data/models`. |
| `--api-model MODEL_ID` | Write the answer with an OpenAI API model (see API mode). |
| `--summary-weight W` | Share of the search score from the chunk summaries, 0 to 1 (default 0.7; 0 = text only). |

The output shows the answer, then the sources. A `*` marks each source that the answer cites. One question takes about 4 seconds on an M5 Pro.

## Ask in the browser

```bash
.venv/bin/python webui.py
```

When the terminal shows the address, open http://127.0.0.1:7860 in your browser.

- Type a question, or click an example question.
- Under each answer, a `✓` marks each source that the answer cites. Click a source to read the full passage.
- Each question is separate. BardBot does not remember earlier questions.
- The page runs only on your computer. Other computers cannot open it.
- Press Ctrl+C in the terminal to stop the page.

`webui.py` accepts the same `--top-k`, `--rerank`, `--model`, and `--api-model` options as `bardbot.py`. Use `--port N` to use another port, and `--open` to open the browser automatically.

## API mode (optional)

In API mode, an OpenAI model writes the answer. The search still runs on your computer, and the prompt is the same as for the local LLM. Each question costs money and needs an internet connection.

API mode also uses **query rewriting** and **8 passages** by default. Before the search, the API model describes the scene that answers the question in 2 modern sentences, for example "In Macbeth, Macbeth speaks alone and imagines a dagger floating before him…". The search score is the average of the question score and the description score. So each question makes 2 API calls and takes about 6–7 seconds. Use `--no-rewrite --top-k 5` for the older, faster setup.

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
- `webui.py`: the chat page in the browser.
- `grade_answers.py`: grades saved answers as correct, partial, or wrong with an API model (see Evaluation).
- `inspect_play.py` and `embed_one_chunk.py`: small demo scripts.

## Evaluation

The question sets are in `data/evaluation/`. Each answerable question has a reference quote from the scene that answers it.

| File | Questions | Use |
|---|---|---|
| `retrieval_cases.json` | 5 | Original development questions |
| `retrieval_cases_holdout.json` | 18 | More development questions |
| `retrieval_cases_test.json` | 16 | Test questions. Written before the reranker work. Later used as development questions for the summary work. |
| `retrieval_cases_test2.json` | 15 | Test questions for the summary work. Used one time, for the final measurement. Now development questions. |
| `unanswerable_dev.json`, `unanswerable_test.json` | 4, 3 | Questions that the plays do not answer. The correct reply is "no answer". Both are now development questions. |
| `retrieval_cases_test3.json` | 25 | Test questions for the query-rewriting work. Used one time, for the final measurement. 5 of them quote the play (`"type": "quote"`). |
| `unanswerable_test2.json` | 5 | Unanswerable test questions for the query-rewriting work, used one time: 2 about details that the plays do not give (for example "How old is Romeo?") and 3 about other books or plays. |

The test3 questions were written from the play text only, without the search results or the chunk summaries. Run the test files only one time, after all design choices for the next change.

Commands:

```bash
.venv/bin/python evaluate_retrieval.py data/evaluation/retrieval_cases_test.json
.venv/bin/python evaluate_answers.py
```

`evaluate_answers.py` saves every answer in `data/evaluation/answers.jsonl`. Its numbers only check citations, not if an answer is correct.

To check if the answers are correct, grade them with an API model. This needs an API key and costs less than one cent per answer:

```bash
.venv/bin/python grade_answers.py --model gpt-6.1-terra data/evaluation/answers_api_summaries.jsonl
```

For each answer, the grader sees the question, the reference passage, and the answer, but not the passages that BardBot got. It gives one grade (correct, partial, or wrong) and a short reason. The grades go to `data/evaluation/grades/`. Read some reasons by hand before you trust the numbers.

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

These results use the "Evidence first" prompt in `answer.py`, on the 43 development questions (39 answerable, 4 unanswerable). With the same search, the same passages went to both models.

| | Phi-4-mini, old search | gpt-6.1-terra, old search | Phi-4-mini, summaries | gpt-6.1-terra, summaries |
|---|---|---|---|---|
| Reference passage among the 5 | 20 | 20 | 29 | 29 |
| **Answer cites the reference** | 11 | 20 | 18 | **28** |
| "No answer" replies to answerable questions | 2 | 4 | 0 | 0 |
| Unanswerable questions handled correctly | 4 of 4 | 4 of 4 * | 2 of 4 | 4 of 4 * |
| Average time per question | 3.6 s | 3.9 s | 4.4 s | 2.9 s |

\* The script counts 3 of 4. But the fourth answer ("What does Macbeth eat for breakfast?") said clearly that the passages do not say this, so it was correct.

- With the old prompt, Phi-4-mini replied "no answer" to 13 of the 16 test questions. The "Evidence first" prompt fixed this for both models.
- With the summaries and gpt-6.1-terra, 28 of 39 answers cite the reference passage. In the one case where it did not, it cited the next chunk of the same scene, and the answer was correct.
- With the summaries, Phi-4-mini answered 2 questions about other plays (Julius Caesar and The Tempest) from its own knowledge. The new search found loosely related passages, and the small model did not refuse.

### Final test (answers)

One run on the unseen test questions, after all design choices: 15 answerable (`retrieval_cases_test2.json`) and 3 unanswerable (`unanswerable_test.json`). Both models used the summary search (W = 0.7).

| Test (15 + 3) | Phi-4-mini | gpt-6.1-terra |
|---|---|---|
| Reference passage among the 5 | 12 | 12 |
| **Answer cites the reference** | 10 | **12** |
| "No answer" replies to answerable questions | 2 | 0 |
| Unanswerable questions handled correctly | 1 of 3 * | 3 of 3 |
| Average time per question | 4.1 s | 3.0 s |

\* The script counts 2 of 3. But Phi-4-mini answered the Othello question from its own knowledge, and for the King Lear question it repeated one line until the token limit.

- The test result matches the dev result: gpt-6.1-terra cited the reference in 12 of 15 test questions (80%) and 28 of 39 dev questions (72%).
- gpt-6.1-terra cited the reference every time that the search found it. Its 3 misses are search misses. For 2 of them, it still gave a correct partial answer from nearby passages, and for the third it said what the passages do not show.
- Phi-4-mini once replied "no answer" when the reference was the first passage, and it answered a question about another play.

### Correctness grades

`grade_answers.py` with gpt-6.1-terra as the grader. All of these questions are development questions now (test2 was used for the final test above).

| Run | Answerable: correct / partial / wrong | Unanswerable: correct / partial / wrong |
|---|---|---|
| Phi-4-mini, old search (43 dev) | 9 / 11 / 19 | 4 / 0 / 0 |
| gpt-6.1-terra, old search (43 dev) | 23 / 2 / 14 | 4 / 0 / 0 |
| Phi-4-mini, summaries (43 dev) | 10 / 9 / 20 | 2 / 0 / 2 |
| **gpt-6.1-terra, summaries (43 dev)** | **30 / 5 / 4** | **4 / 0 / 0** |
| Phi-4-mini, summaries (test2 + 3) | 5 / 5 / 5 | 1 / 1 / 1 |
| gpt-6.1-terra, summaries (test2 + 3) | 11 / 3 / 1 | 3 / 0 / 0 |

- **gpt-6.1-terra with the summaries is correct for 30 of 39 dev questions** (old search: 23), and wrong for only 4.
- **The citation numbers made Phi-4-mini look better than it is.** With the summaries, 18 of its answers cite the reference passage, but only 10 are correct. It often cites the correct passage and then gets the facts wrong (for example, it says that Gertrude dies from Laertes' sword). The better search hardly changed its correct answers (9 → 10).
- **Hand check of the grader:** 26 grades were compared with a careful human reading. 25 agree fully, and the last one is a little strict ("partial" for a correct answer without one detail). The grader gave "wrong" and "partial" to its own model's answers too, so no strong self-bias showed. It also corrected one error of `evaluate_answers.py`: Phi-4-mini's Othello answer is "partial", not a correct refusal.

### Query rewriting and 8 passages (API mode default)

All 61 development questions (54 answerable, 7 unanswerable), gpt-6.1-terra for the rewrite, the answer, and the grade.

Search only, 54 answerable dev questions:

| Search | Top 5 | Top 8 | MRR |
|---|---|---|---|
| Question only | 41 | 43 | 0.615 |
| Rewrite only | 49 | 52 | 0.791 |
| **Average of question and rewrite (default)** | **49** | **51** | 0.777 |

The average is almost as good as the rewrite alone. It keeps the exact words of the question, which helps questions that quote the play.

Answers, graded:

| Setup | Answerable: correct / partial / wrong (of 54) | Unanswerable correct (of 7) | Time per question |
|---|---|---|---|
| No rewrite, 5 passages | 41 / 8 / 5 | 7 | 3.0 s |
| Rewrite, 5 passages | 51 / 3 / 0 | 7 | 6.8 s |
| **Rewrite, 8 passages (default)** | **53 / 1 / 0** | **7** | 6.5 s |

- The one "partial" answer (why Macbeth feels safe in his castle) gave the castle's strength but missed the Birnam Wood prophecy.
- A hand check of the answers that changed to "correct" agreed with the grader. The grader can vary by about one grade between runs.
- For questions about other plays, the rewrite only says what the question asks about. So the search finds nothing that fits, and the answer model still refuses.
- These are development results. The final test is below.

### Final test of query rewriting (test3)

One run on the 30 test3 questions (25 answerable, 5 unanswerable), after all design choices. Nothing was tuned on these questions, so both rows are fair results.

| Setup | Reference among the passages | Answerable: correct / partial / wrong (of 25) | Quote questions correct (of 5) | Unanswerable correct (of 5) | Time |
|---|---|---|---|---|---|
| No rewrite, 5 passages | 15 | 19 / 1 / 5 | 1 | 5 | 3.1 s |
| **Rewrite, 8 passages (default)** | **23** | **24 / 0 / 1** | **4** | **5** | 6.6 s |

- The test result matches the dev result (53 of 54 correct on dev, 24 of 25 on test).
- **Query rewriting helps quote questions most** (1 → 4 of 5). The question alone often does not find the scene of a quote, but the rewrite names the scene.
- The one wrong answer ("Why does Juliet say 'O happy dagger'?") is a search miss. The rewrite described the correct scene, but the chunk with the line was not among the 8 passages.
- The hard unanswerable questions ("How old is Romeo?", "What is the name of Hamlet's dog?") got correct replies: the answers say that the plays do not give this, and mention only related facts.
- A hand check of the wrong answer, the quote questions, and all unanswerable questions agreed with the grader.

## Known limits

- **Retrieval still misses some reference passages:** 10 of 39 dev questions and 3 of 15 test2 questions are not in the top 5.
- **The summaries come from an API model.** To build them again, you need an API key. The summaries can contain small errors or knowledge about the play from outside the passage.
- **Small local model errors.** Phi-4-mini sometimes adds facts that are not in the passages, mixes up characters, or quotes lines longer than 15 words. With the new search, it also answered 2 of 4 questions about other plays from its own knowledge.
- **Small question sets.** 54 retrieval questions in total, so one or two questions change a number a lot.
- **API mode is slower and makes 2 calls per question** (rewrite and answer), about 6–7 seconds. The rewrite uses the model's own knowledge of the plays to name the scene.
- **The grader is an API model, the same model as the API answer model.** A hand check found no strong self-bias, but a different grader model would be a safer test.
- **All test questions are now used**, test3 and `unanswerable_test2.json` too. The next design change needs new, unseen test questions.

## Ideas for next steps

- Write new test questions before the next design change.
- Refuse questions about other plays in code, before the LLM, for example with a score threshold.
- Use a larger local model (for example a 7B model). This needs about 15 GB of memory.
- Write more evaluation questions, especially questions that quote the play.
- Try a stronger embedding model, or smaller chunks.

## Licenses of the models

- `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base`: MIT
- `microsoft/Phi-4-mini-instruct`: MIT
- `data/chunks/summaries.jsonl` was written with the OpenAI API model `gpt-6.1-terra`. OpenAI's terms apply to this output.
