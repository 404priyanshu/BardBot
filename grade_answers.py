"""Grade BardBot answers as correct, partial, or wrong with an OpenAI API model.

evaluate_answers.py checks only which passage an answer cites. This script
checks if the answer is right. For each answer, the grader sees the question,
the reference passage (the chunk that holds the reference quote), and the
answer. It does not see the passages that BardBot got.

    python grade_answers.py --model gpt-6.1-terra data/evaluation/answers_api_summaries.jsonl
    python grade_answers.py --model gpt-6.1-terra data/evaluation/answers_*.jsonl

Needs OPENAI_API_KEY (environment or .env). Costs money. The grades go to
data/evaluation/grades/<answer file name>.jsonl.

Note: when the grader and the answer model are the same model, the grader
can favour its own answers. Read some grades by hand before you trust them.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from answer import load_api_client
from evaluate_retrieval import find_expected_indices

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
CASES_FOLDER = Path("data/evaluation")
GRADES_FOLDER = Path("data/evaluation/grades")
GRADES = ("correct", "partial", "wrong")
MAX_OUTPUT_TOKENS = 4000

INSTRUCTIONS = """You grade answers from a question-answering bot about three Shakespeare plays: Hamlet, Macbeth, and Romeo and Juliet. The bot must answer only from passages of these plays.

You get the question, a reference passage when the question has one, and the bot's answer. First write a short reason (one or two sentences), then give one grade.

If the question has a reference passage (an answerable question):
- correct: the answer gives the main point that the reference passage supports, and it has no important false statement. Extra correct details are fine.
- partial: the answer has some correct, relevant content but misses the main point, or it says that it can answer only part of the question, or it has a small error.
- wrong: the main point is false or about something else, or the answer says the question cannot be answered.

If the question has no reference passage (the plays do not answer it):
- correct: the answer says that the passages do not answer the question, and it gives no answer from outside knowledge. It may mention related facts from the plays.
- partial: the answer says that the passages do not answer the question, but it also gives an answer or a guess from outside knowledge.
- wrong: the answer answers the question anyway, or the text is broken (for example the same line again and again).

Judge only if the content is right. Do not judge the format, the citation numbers, or the quotation length. Use the reference passage and your knowledge of the play."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "grade": {"type": "string", "enum": list(GRADES)},
    },
    "required": ["reason", "grade"],
    "additionalProperties": False,
}


def load_cases():
    """Return every evaluation question by ID, from all question files."""
    cases = {}
    for path in sorted(CASES_FOLDER.glob("*.json")):
        for case in json.loads(path.read_text(encoding="utf-8")):
            if case["id"] in cases:
                raise ValueError(f"Question ID {case['id']} is used twice.")
            cases[case["id"]] = case
    return cases


def reference_text(case, chunks):
    if not case.get("answerable", True):
        return None
    indices = sorted(find_expected_indices(case, chunks))
    return "\n\n".join(chunks[index]["embed_text"] for index in indices)


def grade_one(client, model, record, reference):
    if reference:
        context = (f"Reference quote: {record['reference_quote']}\n\n"
                   f"Reference passage:\n{reference}")
    else:
        context = "Reference passage: none. The plays do not answer this question."
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=f"Question: {record['question']}\n\n{context}\n\nBot answer:\n{record['answer']}",
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text={"format": {"type": "json_schema", "name": "grade", "schema": SCHEMA, "strict": True}},
    )
    result = json.loads(response.output_text)
    if result["grade"] not in GRADES:
        raise ValueError(f"Unexpected grade: {result['grade']}")
    return result


def count(records, answerable):
    group = [r for r in records if r["answerable"] == answerable]
    counts = {grade: sum(r["grade"] == grade for r in group) for grade in GRADES}
    return len(group), counts


def main():
    parser = argparse.ArgumentParser(description="Grade BardBot answers with an OpenAI API model.")
    parser.add_argument("answer_files", nargs="+", help="Answer files from evaluate_answers.py.")
    parser.add_argument("--model", required=True, help="OpenAI model ID of the grader.")
    parser.add_argument("--workers", type=int, default=8, help="Requests at the same time.")
    args = parser.parse_args()

    with CHUNKS_PATH.open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream]
    cases = load_cases()
    client = load_api_client()
    GRADES_FOLDER.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in args.answer_files:
        path = Path(name)
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        references = [reference_text(cases[r["id"]], chunks) for r in records]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(
                lambda pair: grade_one(client, args.model, *pair), zip(records, references)
            ))

        graded = []
        output_path = GRADES_FOLDER / path.name
        with output_path.open("w", encoding="utf-8") as output:
            for record, result in zip(records, results):
                row = {
                    "id": record["id"],
                    "set": record["set"],
                    "answerable": record["answerable"],
                    "llm": record.get("llm"),
                    "grader": args.model,
                    "grade": result["grade"],
                    "reason": result["reason"],
                }
                graded.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

        n_ans, ans = count(graded, True)
        n_una, una = count(graded, False)
        rows.append((path.name, n_ans, ans, n_una, una))
        print(f"Graded {len(graded)} answers from {path} -> {output_path}")

    print(f"\nGrader: {args.model}")
    print(f"{'Answer file':42} {'Answerable: correct / partial / wrong':40} Unanswerable: correct / partial / wrong")
    for file_name, n_ans, ans, n_una, una in rows:
        answerable = f"{ans['correct']:2} / {ans['partial']:2} / {ans['wrong']:2}  (of {n_ans})" if n_ans else "-"
        unanswerable = f"{una['correct']} / {una['partial']} / {una['wrong']}  (of {n_una})" if n_una else "-"
        print(f"{file_name:42} {answerable:40} {unanswerable}")
    print("\nRead some grades and reasons by hand before you trust these numbers.")


if __name__ == "__main__":
    main()
