"""Ask BardBot a question about Hamlet, Macbeth, or Romeo and Juliet.

Usage:
    python bardbot.py "Why does Hamlet fear what happens after death?"
    python bardbot.py            # chat loop: one question at a time
"""
import argparse
import os
import time

# All models are in data/models already (see download_models.py).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from answer import (  # noqa: E402
    LLM_NAME,
    generate_answer,
    generate_answer_api,
    labelled_passage,
    load_api_client,
    load_generator,
    rewrite_query,
)
from search_chunks import (  # noqa: E402
    DEFAULT_SUMMARY_WEIGHT,
    RERANK_CANDIDATES,
    chunk_location,
    chunk_speakers,
    load_index,
    load_reranker,
    load_summary_index,
    search,
)


# Passages sent to the LLM. On the 54 dev questions with query rewriting,
# gpt-6.1-terra was correct for 53 with 8 passages and 51 with 5. The local
# model keeps 5, because more text makes its prompt longer and slower.
DEFAULT_TOP_K_API = 8
DEFAULT_TOP_K_LOCAL = 5


class BardBot:
    def __init__(self, rerank=False, candidates=RERANK_CANDIDATES, top_k=None, model_name=LLM_NAME,
                 api_model=None, summary_weight=DEFAULT_SUMMARY_WEIGHT, rewrite=None):
        """api_model: an OpenAI model ID. When set, the API writes the answers
        and the local LLM is not loaded. The search always runs locally.
        summary_weight: share of the search score from the chunk summaries.
        rewrite: let the API model describe the scene before the search
        (answer.rewrite_query). Default: on in API mode, off otherwise.
        top_k: passages sent to the LLM. Default: 8 in API mode, 5 otherwise."""
        if top_k is None:
            top_k = DEFAULT_TOP_K_API if api_model else DEFAULT_TOP_K_LOCAL
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not 0 <= summary_weight <= 1:
            raise ValueError("summary_weight must be between 0 and 1.")
        if rewrite is None:
            rewrite = bool(api_model)
        if rewrite and not api_model:
            raise ValueError("Query rewriting needs API mode (--api-model).")
        self.candidates = candidates
        self.top_k = top_k
        self.summary_weight = summary_weight
        self.rewrite = rewrite
        self.api_model = api_model
        self.llm_name = api_model or model_name
        # Check the API settings first, before the slow model loads.
        self.api_client = load_api_client() if api_model else None
        self.embedder, self.embeddings, self.chunks = load_index()
        self.reranker = load_reranker() if rerank else None
        self.summary_embeddings = load_summary_index(self.chunks) if summary_weight else None
        if api_model:
            self.tokenizer = self.llm = None
        else:
            self.tokenizer, self.llm = load_generator(model_name)

    def retrieve(self, question, rewrite=None):
        ranked_indices, _ = search(
            question, self.embedder, self.embeddings, self.chunks,
            self.reranker, self.candidates,
            self.summary_embeddings, self.summary_weight, rewrite,
        )
        return ranked_indices[:self.top_k]

    def ask(self, question):
        question = question.strip()
        if not question:
            raise ValueError("Please ask a non-empty question.")
        start = time.perf_counter()
        rewrite = rewrite_query(question, self.api_client, self.api_model) if self.rewrite else None
        indices = self.retrieve(question, rewrite)
        passages = [self.chunks[index] for index in indices]
        if self.api_model:
            result = generate_answer_api(question, passages, self.api_client, self.api_model)
        else:
            result = generate_answer(question, passages, self.tokenizer, self.llm)
        result.update({
            "question": question,
            "rewrite": rewrite,
            "passage_indices": indices,
            "passages": passages,
            "seconds": time.perf_counter() - start,
        })
        return result


def print_result(result, show_passages=False):
    print("\nAnswer:\n" + result["answer"])
    if result["hit_token_limit"]:
        print("(The answer reached the token limit and may be cut off.)")

    print("\nSources:")
    for number, chunk in enumerate(result["passages"], start=1):
        marker = "*" if number in result["cited"] else " "
        speakers = ", ".join(chunk_speakers(chunk)) or "no speaker listed"
        print(f" {marker}[{number}] {chunk_location(chunk)} | {speakers}")
        if show_passages:
            print("     " + labelled_passage(chunk).replace("\n", "\n     "))
    print("(* = cited in the answer)")
    print(f"Time: {result['seconds']:.1f} s")


def main():
    parser = argparse.ArgumentParser(description="Answer questions about three Shakespeare plays.")
    parser.add_argument("question", nargs="?", help="Leave out to start a chat loop.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Passages sent to the LLM (default: 8 in API mode, 5 otherwise).")
    parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=False,
        help="Re-score search results with the reranker (default: off; it did not "
             "help on the dev questions). Needs: download_models.py --reranker",
    )
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    parser.add_argument("--show-passages", action="store_true", help="Print the full passages.")
    parser.add_argument("--model", default=LLM_NAME, help="Local LLM in data/models.")
    parser.add_argument(
        "--api-model", metavar="MODEL_ID",
        help="Write answers with this OpenAI API model instead of the local LLM. "
             "Needs OPENAI_API_KEY. Costs money for each question.",
    )
    parser.add_argument(
        "--summary-weight", type=float, default=DEFAULT_SUMMARY_WEIGHT,
        help=f"Share of the search score from the chunk summaries, 0 to 1 "
             f"(default {DEFAULT_SUMMARY_WEIGHT}; 0 = text only).",
    )
    parser.add_argument(
        "--rewrite", action=argparse.BooleanOptionalAction, default=None,
        help="Let the API model describe the scene before the search "
             "(default: on with --api-model; needs --api-model).",
    )
    args = parser.parse_args()

    if args.question is not None and not args.question.strip():
        parser.error("Please ask a non-empty question.")

    print("Loading models...")
    try:
        bot = BardBot(args.rerank, args.candidates, args.top_k, args.model, args.api_model,
                      args.summary_weight, args.rewrite)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print("Answers by:", bot.llm_name)

    if args.question:
        print_result(bot.ask(args.question), args.show_passages)
        return

    print("Ask about Hamlet, Macbeth, or Romeo and Juliet. Type 'quit' or press Ctrl+D to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print_result(bot.ask(question), args.show_passages)


if __name__ == "__main__":
    main()
