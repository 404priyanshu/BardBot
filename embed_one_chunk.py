import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

path = Path("data/chunks/chunks.jsonl")
with path.open(encoding="utf-8") as stream:
    first_line = next(stream, "")

if not first_line.strip():
    raise ValueError("No first chunk found to embed.")

chunk = json.loads(first_line)

model = SentenceTransformer(
    MODEL_NAME,
    cache_folder="data/models",
    device="cpu",
)

# Check the actual input before encode can silently truncate it.
token_ids = model.tokenizer(chunk["embed_text"], truncation=False)["input_ids"]
if len(token_ids) > model.max_seq_length:
    raise ValueError("The first chunk exceeds the model's token limit.")

embedding = model.encode(
    chunk["embed_text"],
    normalize_embeddings=True,
    show_progress_bar=False,
)

print("Model:", MODEL_NAME)
print("Location:", chunk["title"], chunk["act"], chunk["scene"])
print("Scene word range:", chunk["start_word"], "to", chunk["end_word"])
print("Input tokens:", len(token_ids))
print("Token limit:", model.max_seq_length)
print("Embedding shape:", embedding.shape)
print("First 8 values:", [round(float(value), 6) for value in embedding[:8]])
