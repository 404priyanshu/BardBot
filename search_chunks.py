import argparse
import json

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANKER_NAME = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 30


def best_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"

def load_index():
    with np.load("data/embeddings/chunk_embeddings.npz", allow_pickle=False) as saved:
        embeddings = saved["embeddings"]
        chunks = [json.loads(record) for record in saved["chunks"]]
        if saved["model_name"].item() != MODEL_NAME:
            raise ValueError("Saved embeddings use a different model.")
        if not saved["normalized"].item():
            raise ValueError("This search expects normalized chunk embeddings.")

    if not chunks:
        raise ValueError("No saved chunks found to search.")

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder="data/models",
        device="cpu",
        local_files_only=True,
    )

    if embeddings.shape != (len(chunks), model.get_embedding_dimension()):
        raise ValueError("Embedding shape does not match the chunks and model.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Saved embeddings contain NaN or infinity.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6):
        raise ValueError("Saved chunk vectors are not unit length.")

    return model, embeddings, chunks


def rank_chunks(question, model, embeddings):
    question = question.strip()
    if not question:
        raise ValueError("Please provide a non-empty question.")

    query_text = QUERY_INSTRUCTION + question
    query_tokens = model.tokenizer(query_text, truncation=False)["input_ids"]
    if len(query_tokens) > model.max_seq_length:
        raise ValueError("Question plus retrieval instruction exceeds the model's token limit.")

    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # For unit-length vectors, dot product equals cosine similarity.
    scores = embeddings @ query_embedding
    ranked_indices = np.argsort(-scores, kind="stable")
    return scores, ranked_indices, len(query_tokens)


def load_reranker():
    return CrossEncoder(
        RERANKER_NAME,
        cache_folder="data/models",
        device=best_device(),
        local_files_only=True,
        max_length=512,
    )


def rerank_chunks(question, ranked_indices, chunks, reranker, candidates=RERANK_CANDIDATES):
    """Re-score the top candidates from the first search with a cross-encoder.

    A cross-encoder reads the question and the passage together, which often
    judges relevance better than comparing two separate vectors, but it is too
    slow to run on every chunk. On the dev questions it did NOT beat plain
    search (MRR 0.34 vs 0.39 with 30 candidates), so it is off by default. Returns all indices: the reranked candidates first,
    then the rest in their original order, and the reranker scores by index.
    """
    if candidates <= 0:
        raise ValueError("Candidates must be positive.")

    top = [int(index) for index in ranked_indices[:candidates]]
    pairs = [(question, chunks[index]["embed_text"]) for index in top]
    pair_scores = reranker.predict(pairs, show_progress_bar=False)

    order = np.argsort(-np.asarray(pair_scores), kind="stable")
    reranked = [top[position] for position in order]
    rest = [int(index) for index in ranked_indices[candidates:]]
    scores = {index: float(score) for index, score in zip(top, pair_scores)}
    return reranked + rest, scores


def search(question, model, embeddings, chunks, reranker=None, candidates=RERANK_CANDIDATES):
    """Return ranked chunk indices and a score for each top index."""
    scores, ranked_indices, _ = rank_chunks(question, model, embeddings)
    if reranker is None:
        return [int(index) for index in ranked_indices], {
            int(index): float(scores[index]) for index in ranked_indices
        }
    return rerank_chunks(question, ranked_indices, chunks, reranker, candidates)


def chunk_speakers(chunk):
    return list(dict.fromkeys(
        speaker
        for span in chunk["speech_spans"]
        for speaker in span["speakers"]
    ))


def chunk_location(chunk):
    return f"{chunk['title']} | {chunk['act']} | {chunk['scene'] or chunk['section'].upper()}"


def main():
    parser = argparse.ArgumentParser(description="Find three relevant Shakespeare passages.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Why does Hamlet fear what happens after death?",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-score the top results with the reranker (default: off).",
    )
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    args = parser.parse_args()
    question = args.question.strip()
    if not question:
        parser.error("Please provide a non-empty question.")

    model, embeddings, chunks = load_index()
    reranker = load_reranker() if args.rerank else None
    try:
        ranked_indices, scores = search(
            question, model, embeddings, chunks, reranker, args.candidates
        )
    except ValueError as error:
        parser.error(str(error))

    score_name = "reranker score" if args.rerank else "cosine similarity"
    print("Question:", question)
    print("Chunks searched:", len(chunks))
    print("Reranker:", f"on ({args.candidates} candidates)" if args.rerank else "off")

    for rank, index in enumerate(ranked_indices[:3], start=1):
        chunk = chunks[index]
        speakers = chunk_speakers(chunk)
        print(f"\nResult {rank} | {score_name}: {scores[index]:.4f}")
        print("Title:", chunk["title"])
        print("Source:", chunk["source_file"])
        print("Location:", chunk["act"], "|", chunk["scene"] or chunk["section"].upper())
        print("Chunk number within section:", chunk["chunk_number"])
        print("Scene/prologue word range:", chunk["start_word"], "to", chunk["end_word"])
        print("Speaker labels in chunk:", ", ".join(speakers) or "Not specified in source")
        print("Passage:", chunk["text"])


if __name__ == "__main__":
    main()
