import json
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOKEN_LIMIT = 512

path = Path("data/chunks/chunks.jsonl")
with path.open(encoding="utf-8") as stream:
    chunks = [json.loads(line) for line in stream]

if not chunks:
    raise ValueError("No chunks found to check.")

# Download the tokenizer files once, then reuse the local cache.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="data/models")
token_counts = []
oversized = []

for number, chunk in enumerate(chunks, start=1):
    encoded = tokenizer(
        chunk["embed_text"],
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )
    count = len(encoded["input_ids"])
    token_counts.append(count)

    if count > TOKEN_LIMIT:
        oversized.append((number, count))

print("Model:", MODEL_NAME)
print("Chunks checked:", len(chunks))
print("Token limit:", TOKEN_LIMIT)
print("Shortest chunk:", min(token_counts), "tokens")
print("Median chunk:", median(token_counts), "tokens")
print("Longest chunk:", max(token_counts), "tokens")
print("Chunks over limit:", len(oversized))
print("First chunk words:", len(chunks[0]["text"].split()))
print("First chunk tokens:", token_counts[0])

for number, count in oversized:
    chunk = chunks[number - 1]
    print("Over limit:", number, chunk["title"], chunk["act"], chunk["scene"], count)
