import argparse
import json
from pathlib import Path
from statistics import median

from search_chunks import (
    DEFAULT_SUMMARY_WEIGHT,
    RERANK_CANDIDATES,
    load_index,
    load_reranker,
    load_summary_index,
    search,
)


def find_expected_indices(case, chunks):
    # Accept every overlapping chunk that contains the reference quote.
    expected = {
        index
        for index, chunk in enumerate(chunks)
        if chunk["source_file"] == case["source_file"]
        and chunk["act"] == case["act"]
        and chunk["scene"] == case["scene"]
        and case["anchor"].casefold() in chunk["text"].casefold()
    }
    if not expected:
        raise ValueError(f"Reference passage not found for {case['id']}.")
    return expected


def first_relevant_rank(ranked_indices, expected):
    return next(
        rank
        for rank, index in enumerate(ranked_indices, start=1)
        if index in expected
    )


def main():
    parser = argparse.ArgumentParser(description="Measure retrieval hit rates.")
    parser.add_argument(
        "cases",
        nargs="*",
        default=["data/evaluation/retrieval_cases.json"],
        help="One or more JSON files of evaluation questions.",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Re-score the first-stage results with the reranker.",
    )
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    parser.add_argument("--quiet", action="store_true", help="Print only the summary.")
    parser.add_argument(
        "--summary-weight", type=float, default=DEFAULT_SUMMARY_WEIGHT,
        help="Share of the score from the chunk summaries, 0 to 1 (0 = text only).",
    )
    args = parser.parse_args()
    if not 0 <= args.summary_weight <= 1:
        parser.error("--summary-weight must be between 0 and 1.")

    cases = []
    for name in args.cases:
        path = Path(name)
        file_cases = json.loads(path.read_text(encoding="utf-8"))
        print(f"Cases file: {path} ({len(file_cases)} questions)")
        cases.extend(file_cases)
    if not cases:
        raise ValueError("No evaluation questions found.")

    model, embeddings, chunks = load_index()
    reranker = load_reranker() if args.rerank else None
    summary_embeddings = load_summary_index(chunks) if args.summary_weight else None
    references = [find_expected_indices(case, chunks) for case in cases]
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    ranks = []

    print("Evaluation questions:", len(cases))
    print("Chunks searched per question:", len(chunks))
    print("Reranker:", f"on ({args.candidates} candidates)" if args.rerank else "off")
    print("Summary weight:", args.summary_weight)
    print("Match rule: source + act + scene + reference quote.")

    for case, expected in zip(cases, references):
        # Reference metadata is for scoring only, never for filtering search.
        ranked_indices, scores = search(
            case["question"], model, embeddings, chunks, reranker, args.candidates,
            summary_embeddings, args.summary_weight,
        )
        rank = first_relevant_rank(ranked_indices, expected)
        ranks.append(rank)
        for k in hits:
            hits[k] += int(rank <= k)

        if args.quiet:
            continue
        best_index = ranked_indices[rank - 1]
        score = scores.get(best_index)
        score_text = f"{score:.4f}" if score is not None else "not reranked"
        print(f"\n{case['id']}: {case['question']}")
        print("Expected:", case["source_file"], "|", case["act"], "|", case["scene"])
        print("Reference quote:", case["anchor"])
        print("Matching reference chunks:", len(expected))
        print(f"First reference rank: {rank} | score: {score_text}")
        print("Top 1 / 3 / 5 / 10:", " / ".join(
            "HIT" if rank <= k else "MISS" for k in hits
        ))

    print("\nSummary (reference-passage hit rate, not answer accuracy):")
    for k, count in hits.items():
        print(f"Hit@{k}: {count}/{len(cases)} ({count / len(cases):.0%})")
    print(f"Median rank: {median(ranks)}")
    print(f"MRR: {sum(1 / rank for rank in ranks) / len(ranks):.3f}")
    print("Ranks:", " ".join(str(rank) for rank in ranks))
    print("Small hand-picked question sets; other relevant passages may not be labelled.")


if __name__ == "__main__":
    main()
