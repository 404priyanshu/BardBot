"""Write a short modern-English summary of each chunk with an OpenAI API model.

The questions are in modern English, but the chunks are in Early Modern
English. A summary in modern words gives the search a second way to match a
question to a chunk (see embed_summaries.py).

    python summarize_chunks.py --model gpt-6.1-terra --limit 5   # try 5 chunks
    python summarize_chunks.py --model gpt-6.1-terra             # all chunks

Needs OPENAI_API_KEY (environment or .env). Costs money. The script continues
from data/chunks/summaries.jsonl, so it skips chunks that already have a summary.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from answer import load_api_client
from search_chunks import chunk_key

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
SUMMARIES_PATH = Path("data/chunks/summaries.jsonl")
MAX_OUTPUT_TOKENS = 2000

INSTRUCTIONS = """You summarize passages from Shakespeare's plays for a search index.

Write two or three sentences of plain modern English, 40 to 70 words in total. Say who speaks, what happens, and what the characters feel, want, decide, or describe. Use modern words for old ones (for example "kill" for "do the deed").

Describe only what happens in this passage. You may use your knowledge of the play only to identify the people and things that the passage refers to. Do not describe events from other parts of the play.

Reply with the summary only."""


def summarize(client, model, chunk):
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=chunk["embed_text"],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    summary = " ".join(response.output_text.split())
    if not summary:
        raise ValueError(f"Empty summary (status: {response.status})")
    usage = response.usage
    return summary, (usage.input_tokens if usage else 0), (usage.output_tokens if usage else 0)


def main():
    parser = argparse.ArgumentParser(description="Summarize chunks with an OpenAI API model.")
    parser.add_argument("--model", required=True, help="OpenAI model ID, for example gpt-6.1-terra.")
    parser.add_argument("--limit", type=int, default=None, help="Summarize at most N new chunks.")
    parser.add_argument("--workers", type=int, default=8, help="Requests at the same time.")
    args = parser.parse_args()

    with CHUNKS_PATH.open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream]
    keys = [chunk_key(chunk) for chunk in chunks]
    if len(set(keys)) != len(keys):
        raise ValueError("Two chunks have the same key.")

    done = set()
    if SUMMARIES_PATH.exists():
        with SUMMARIES_PATH.open(encoding="utf-8") as stream:
            done = {json.loads(line)["key"] for line in stream}
    todo = [(key, chunk) for key, chunk in zip(keys, chunks) if key not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"Chunks: {len(chunks)} | already summarized: {len(done)} | to do now: {len(todo)}")
    if not todo:
        return

    client = load_api_client()
    lock = threading.Lock()
    totals = {"input": 0, "output": 0, "failed": 0}

    with SUMMARIES_PATH.open("a", encoding="utf-8") as output, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(summarize, client, args.model, chunk): (key, chunk)
                   for key, chunk in todo}
        for number, future in enumerate(as_completed(futures), start=1):
            key, chunk = futures[future]
            try:
                summary, input_tokens, output_tokens = future.result()
            except Exception as error:  # Keep going; a new run retries the failed chunks.
                totals["failed"] += 1
                print(f"{number}/{len(todo)} FAILED {key}: {type(error).__name__}: {error}")
                continue
            record = {"key": key, "model": args.model, "summary": summary}
            with lock:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                totals["input"] += input_tokens
                totals["output"] += output_tokens
            if args.limit is not None and args.limit <= 10:
                print(f"\n{chunk['title']} | {chunk['act']} | {chunk['scene'] or chunk['section']}")
                print("Summary:", summary)
            elif number % 25 == 0:
                print(f"{number}/{len(todo)} done", flush=True)

    print(f"\nNew summaries: {len(todo) - totals['failed']} | failed: {totals['failed']}")
    print(f"Tokens: {totals['input']} input, {totals['output']} output")
    if totals["failed"]:
        print("Run the same command again to retry the failed chunks.")


if __name__ == "__main__":
    main()
