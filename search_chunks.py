import argparse
import json

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANKER_NAME = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 30
SUMMARY_INDEX_PATH = "data/embeddings/summary_embeddings.npz"
# Share of the score that comes from the summary vector (0 = text only).
# Chosen on the 39 dev questions: Hit@5 went from 20 to 29 (weights 0.5-1.0
# were close). 0.7 keeps some text score for questions that quote the play.
DEFAULT_SUMMARY_WEIGHT = 0.7


def chunk_key(chunk):
    """A stable ID for a chunk, independent of its position in the file."""
    return "|".join(str(chunk[field]) for field in
                    ("source_file", "act", "section", "scene", "chunk_number", "start_word"))


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


def load_summary_index(chunks, path=SUMMARY_INDEX_PATH):
    """Load the summary vectors (embed_summaries.py) in the same order as chunks."""
    with np.load(path, allow_pickle=False) as saved:
        embeddings = saved["embeddings"]
        keys = [str(key) for key in saved["keys"]]
        if saved["model_name"].item() != MODEL_NAME:
            raise ValueError("Saved summary embeddings use a different model.")
    if keys != [chunk_key(chunk) for chunk in chunks]:
        raise ValueError("Summary embeddings do not match the chunks. Run embed_summaries.py again.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6):
        raise ValueError("Saved summary vectors are not unit length.")
    return embeddings


def score_query(text, model, embeddings, summary_embeddings=None, summary_weight=0.0):
    """Score every chunk for one query text. Returns the scores and the token count."""
    query_text = QUERY_INSTRUCTION + text
    query_tokens = model.tokenizer(query_text, truncation=False)["input_ids"]
    if len(query_tokens) > model.max_seq_length:
        raise ValueError("Query plus retrieval instruction exceeds the model's token limit.")

    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # For unit-length vectors, dot product equals cosine similarity.
    scores = embeddings @ query_embedding
    if summary_weight:
        if summary_embeddings is None:
            raise ValueError("A summary weight needs the summary embeddings.")
        summary_scores = summary_embeddings @ query_embedding
        scores = (1 - summary_weight) * scores + summary_weight * summary_scores
    return scores, len(query_tokens)


def rank_chunks(question, model, embeddings, summary_embeddings=None, summary_weight=0.0,
                rewrite=None):
    """Rank all chunks for a question.

    rewrite: an optional modern description of the scene (answer.rewrite_query).
    When given, the score is the average of the question score and the
    rewrite score. The question score keeps the exact words, for example a quote.
    """
    question = question.strip()
    if not question:
        raise ValueError("Please provide a non-empty question.")

    scores, token_count = score_query(question, model, embeddings, summary_embeddings,
                                      summary_weight)
    if rewrite and rewrite.strip():
        rewrite_scores, _ = score_query(rewrite.strip(), model, embeddings, summary_embeddings,
                                        summary_weight)
        scores = (scores + rewrite_scores) / 2
    ranked_indices = np.argsort(-scores, kind="stable")
    return scores, ranked_indices, token_count


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


def search(question, model, embeddings, chunks, reranker=None, candidates=RERANK_CANDIDATES,
           summary_embeddings=None, summary_weight=0.0, rewrite=None):
    """Return ranked chunk indices and a score for each top index."""
    scores, ranked_indices, _ = rank_chunks(
        question, model, embeddings, summary_embeddings, summary_weight, rewrite
    )
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
    summary_embeddings = load_summary_index(chunks) if DEFAULT_SUMMARY_WEIGHT else None
    try:
        ranked_indices, scores = search(
            question, model, embeddings, chunks, reranker, args.candidates,
            summary_embeddings, DEFAULT_SUMMARY_WEIGHT,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.rerank:
        score_name = "reranker score"
    elif DEFAULT_SUMMARY_WEIGHT:
        score_name = f"text + summary score (summary weight {DEFAULT_SUMMARY_WEIGHT})"
    else:
        score_name = "cosine similarity"
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
