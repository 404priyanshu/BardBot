import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

path = Path("data/chunks/chunks.jsonl")
with path.open(encoding="utf-8") as stream:
    chunks = [json.loads(line) for line in stream]

if not chunks:
    raise ValueError("No chunks found to embed.")

model = SentenceTransformer(
    MODEL_NAME,
    cache_folder="data/models",
    device="cpu",
    local_files_only=True,
)

# Embed the speaker-labelled text; "text" stays unlabelled for display.
texts = [chunk["embed_text"] for chunk in chunks]
token_counts = []
for number, text in enumerate(texts, start=1):
    token_ids = model.tokenizer(text, truncation=False)["input_ids"]
    token_counts.append(len(token_ids))
    if len(token_ids) > model.max_seq_length:
        raise ValueError(f"Chunk record {number} exceeds the model's token limit.")

embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)

expected_shape = (len(chunks), model.get_embedding_dimension())
if embeddings.shape != expected_shape:
    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
if not np.isfinite(embeddings).all():
    raise ValueError("Embeddings contain NaN or infinity.")
norms = np.linalg.norm(embeddings, axis=1)
if not np.allclose(norms, 1.0, atol=1e-6):
    raise ValueError("Embeddings are not unit-length vectors.")

# Keep full chunk records in the same order as the embedding rows.
chunk_records = np.array([
    json.dumps(chunk, ensure_ascii=False) for chunk in chunks
])
output_folder = Path("data/embeddings")
output_folder.mkdir(parents=True, exist_ok=True)
output_path = output_folder / "chunk_embeddings.npz"
np.savez_compressed(
    output_path,
    embeddings=embeddings,
    chunks=chunk_records,
    model_name=np.array(MODEL_NAME),
    normalized=np.array(True),
)

print("Model:", MODEL_NAME)
print("Chunks embedded:", len(chunks))
print("Longest input:", max(token_counts), "tokens")
print("Token limit:", model.max_seq_length)
print("Embedding matrix shape:", embeddings.shape)
print("Embedding dtype:", embeddings.dtype)
print("Metadata records:", len(chunk_records))
print("Checks passed: expected shape, finite values, unit-length vectors.")
print("Saved:", output_path)
