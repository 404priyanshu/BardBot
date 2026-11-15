"""Run evaluation questions through BardBot and check the citations.

This does not grade whether an answer is correct; read the saved answers for
that. For answerable questions, it checks whether the answer cites the passage
that holds the reference quote, which is a rough sign that it used the right
evidence. For unanswerable questions (no reference quote), the correct reply
is the "no answer" reply.
"""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from answer import NO_ANSWER  # noqa: E402
from bardbot import BardBot  # noqa: E402
from evaluate_retrieval import find_expected_indices  # noqa: E402
from search_chunks import chunk_location  # noqa: E402

DEV_FILES = [
    "data/evaluation/retrieval_cases.json",
    "data/evaluation/retrieval_cases_holdout.json",
    "data/evaluation/retrieval_cases_test.json",
    "data/evaluation/unanswerable_dev.json",
]
TEST_FILES = [
    "data/evaluation/retrieval_cases_test2.json",
    "data/evaluation/unanswerable_test.json",
]


def set_name(path):
    name = path.stem.replace("retrieval_cases", "").strip("_")
    return name or "original"


REFUSAL_PATTERN = re.compile(
    r"passages?(?: provided| given)? (?:do|does) not (?:directly )?"
    r"(?:answer|address|contain|mention|say|describe|include)"
    r"|cannot (?:be )?answer",
    re.IGNORECASE,
)


def is_refusal(answer_text, cited):
    """A refusal says the passages do not answer, and cites no passage.

    The model does not always use the exact NO_ANSWER sentence.
    """
    if NO_ANSWER.rstrip(".").casefold() in answer_text.casefold():
        return True
    return not cited and bool(REFUSAL_PATTERN.search(answer_text))


def print_summary(label, records):
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    print(f"\n{label}:")
    if answerable:
        n = len(answerable)
        count = lambda key: sum(r[key] for r in answerable)  # noqa: E731
        print(f"  Answerable questions: {n}")
        print(f"    Reference passage among the passages: {count('retrieved')}/{n}")
        print(f"    Answers with at least one citation:   {count('cited_any')}/{n}")
        print(f"    Answers that cite the reference:      {count('cited_reference')}/{n}")
        print(f"    'No answer' replies:                  {count('no_answer')}/{n}")
        print(f"    Answers cut off at the token limit:   {count('cut_off')}/{n}")
    if unanswerable:
        n = len(unanswerable)
        print(f"  Unanswerable questions: {n}")
        print(f"    Correct 'no answer' replies:          {sum(r['no_answer'] for r in unanswerable)}/{n}")


def main():
    parser = argparse.ArgumentParser(description="Check BardBot answers and citations.")
    parser.add_argument(
        "cases", nargs="*", default=DEV_FILES + TEST_FILES,
        help="Question files (default: all dev and test files).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Default: data/evaluation/answers.jsonl, or answers_api.jsonl with --api-model.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Passages sent to the LLM.")
    parser.add_argument(
        "--api-model", metavar="MODEL_ID",
        help="Write answers with this OpenAI API model. Needs OPENAI_API_KEY. Costs money.",
    )
    args = parser.parse_args()
    if args.output is None:
        name = "answers_api.jsonl" if args.api_model else "answers.jsonl"
        args.output = f"data/evaluation/{name}"

    cases = []
    for name in args.cases:
        path = Path(name)
        for case in json.loads(path.read_text(encoding="utf-8")):
            case["set"] = set_name(path)
            cases.append(case)
    if not cases:
        raise ValueError("No evaluation questions found.")

    options = {"api_model": args.api_model}
    if args.top_k is not None:
        options["top_k"] = args.top_k
    try:
        bot = BardBot(**options)
    except RuntimeError as error:
        parser.error(str(error))
    print("Answers by:", bot.llm_name)
    print("Passages sent to the LLM:", bot.top_k)
    records = []

    with open(args.output, "w", encoding="utf-8") as output_file:
        for number, case in enumerate(cases, start=1):
            answerable = case.get("answerable", True)
            expected = find_expected_indices(case, bot.chunks) if answerable else set()
            result = bot.ask(case["question"])
            positions = [
                position for position, index in enumerate(result["passage_indices"], start=1)
                if index in expected
            ]
            record = {
                "id": case["id"],
                "set": case["set"],
                "llm": bot.llm_name,
                "answerable": answerable,
                "question": case["question"],
                "reference_quote": case.get("anchor"),
                "answer": result["answer"],
                "cited": result["cited"],
                "reference_positions": positions,
                "sources": [chunk_location(chunk) for chunk in result["passages"]],
                "retrieved": bool(positions),
                "cited_any": bool(result["cited"]),
                "cited_reference": any(position in result["cited"] for position in positions),
                "no_answer": is_refusal(result["answer"], result["cited"]),
                "cut_off": result["hit_token_limit"],
                "seconds": round(result["seconds"], 1),
            }
            records.append(record)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            outcome = (
                f"cited reference={record['cited_reference']}, no answer={record['no_answer']}"
                if answerable else f"correct refusal={record['no_answer']}"
            )
            print(f"{number:2}/{len(cases)} {case['id']}: {outcome}, {result['seconds']:.1f} s")

    for name in dict.fromkeys(r["set"] for r in records):
        print_summary(f"Set '{name}'", [r for r in records if r["set"] == name])
    print_summary("All questions", records)
    print(f"\nAverage time per question: {sum(r['seconds'] for r in records) / len(records):.1f} s")
    print("Saved answers to", args.output, "- read them to judge correctness.")


if __name__ == "__main__":
    main()
