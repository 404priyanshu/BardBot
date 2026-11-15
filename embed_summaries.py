"""Embed the chunk summaries (summarize_chunks.py) for the search.

Each summary gets its own vector, in the same order as the chunks. The search
can then combine the text score and the summary score (--summary-weight).
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from search_chunks import MODEL_NAME, SUMMARY_INDEX_PATH, chunk_key

with Path("data/chunks/chunks.jsonl").open(encoding="utf-8") as stream:
    chunks = [json.loads(line) for line in stream]
with Path("data/chunks/summaries.jsonl").open(encoding="utf-8") as stream:
    summaries = {}
    for line in stream:
        record = json.loads(line)
        summaries[record["key"]] = record["summary"]

keys = [chunk_key(chunk) for chunk in chunks]
missing = [key for key in keys if key not in summaries]
if missing:
    raise ValueError(f"{len(missing)} chunks have no summary. Run summarize_chunks.py first.")

# Same heading line as embed_text (play, act, scene), then the summary.
texts = [chunk["embed_text"].split("\n", 1)[0] + "\n" + summaries[key]
         for chunk, key in zip(chunks, keys)]

model = SentenceTransformer(MODEL_NAME, cache_folder="data/models", device="cpu",
                            local_files_only=True)
longest = max(len(model.tokenizer(text, truncation=False)["input_ids"]) for text in texts)
if longest > model.max_seq_length:
    raise ValueError(f"A summary has {longest} tokens; the limit is {model.max_seq_length}.")

embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
if embeddings.shape != (len(chunks), model.get_embedding_dimension()):
    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
if not np.isfinite(embeddings).all():
    raise ValueError("Embeddings contain NaN or infinity.")

np.savez_compressed(SUMMARY_INDEX_PATH, embeddings=embeddings, keys=np.array(keys),
                    model_name=np.array(MODEL_NAME))
print("Summaries embedded:", len(texts))
print("Longest input:", longest, "tokens (limit", model.max_seq_length, ")")
print("Saved:", SUMMARY_INDEX_PATH)
