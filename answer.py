"""Write an answer from retrieved passages with a local LLM."""
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from search_chunks import best_device

# Hide harmless config warnings from transformers (for example about rope_parameters).
transformers_logging.set_verbosity_error()

LLM_NAME = "microsoft/Phi-4-mini-instruct"
MAX_NEW_TOKENS = 400
# For API models. Reasoning models count their hidden reasoning as output
# tokens too, so this limit is much higher than MAX_NEW_TOKENS.
MAX_API_OUTPUT_TOKENS = 4000
NO_ANSWER = "The passages do not answer this question."

SYSTEM_PROMPT = f"""You answer questions about three Shakespeare plays: Hamlet, Macbeth, and Romeo and Juliet.

The user gives you numbered passages from the plays and a question. The passages are in Shakespeare's Early Modern English. Each line starts with the name of the character who speaks it. The answer is usually in what a character says, and it is often said indirectly or in poetic language.

How to answer:
1. Find the passage that best answers the question. The answer is often only implied by what the characters say.
2. Copy one short line (at most 15 words) from that passage, exactly as written, as evidence.
3. Explain the answer in two to four sentences of plain modern English.
4. End every sentence with the number of the passage that supports it in square brackets, for example [2] for Passage 2.
5. Use only the passages. Do not add facts from outside knowledge. Never copy a whole passage or its heading.

Use this format:
Evidence: "<the short line>" [n]
Answer: <your two to four sentences>

Most questions can be answered from the passages. If the passages support only part of the answer, give that part and say in one sentence what they do not show.

Reply exactly "{NO_ANSWER}", without the format above, only in these two cases:
- The question is not about Hamlet, Macbeth, or Romeo and Juliet.
- No passage has anything to do with the question.

Example:
Question: Why does Macbeth hesitate to kill the king?
Evidence: "He's here in double trust" [1]
Answer: Macbeth fears that the murder will bring punishment in this life and that others will copy his crime against him [1]. He also says Duncan trusts him twice, as his relative and as his host, so he should protect him, not kill him [1]."""


def load_generator(model_name=LLM_NAME):
    device = best_device()
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir="data/models", local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir="data/models", local_files_only=True, dtype=dtype
    ).to(device)
    model.eval()
    return tokenizer, model


def labelled_passage(chunk):
    """Rebuild the chunk text with a speaker label before each speech."""
    words = chunk["text"].split()
    parts = []
    for span in chunk["speech_spans"]:
        start = span["start_word"] - chunk["start_word"]
        end = span["end_word"] - chunk["start_word"] + 1
        text = " ".join(words[start:end])
        label = " and ".join(span["speakers"])
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def passage_heading(chunk):
    location = chunk["scene"] or chunk["section"].upper()
    return " ".join(f"{chunk['title']}, {chunk['act']}, {location}".split())


def build_messages(question, chunks):
    # "Passage n (...)" and not "[n] ...": when the model starts its answer
    # with "[1]", it tends to copy the whole passage that follows that label.
    passages = "\n\n".join(
        f"Passage {number} ({passage_heading(chunk)}):\n{labelled_passage(chunk)}"
        for number, chunk in enumerate(chunks, start=1)
    )
    # Small models follow the task better when the question comes before and
    # after the long passages.
    user_message = (
        f"Question: {question}\n\nPassages:\n\n{passages}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


CITATION_PATTERN = re.compile(
    r"\[(?:Passages?\s*)?(\d+(?:\s*(?:,|and)\s*\d+)*)\]"   # [2], [1, 3], [Passage 2]
    r"|\(Passages?\s*(\d+(?:\s*(?:,|and)\s*\d+)*)\)",      # (Passage 4)
    re.IGNORECASE,
)


def cited_numbers(answer_text, passage_count):
    """Return the valid passage numbers that the answer cites, in order."""
    numbers = []
    for match in CITATION_PATTERN.finditer(answer_text):
        group = match.group(1) or match.group(2)
        for value in re.findall(r"\d+", group):
            number = int(value)
            if 1 <= number <= passage_count and number not in numbers:
                numbers.append(number)
    return numbers


def generate_answer(question, chunks, tokenizer, model, max_new_tokens=MAX_NEW_TOKENS):
    if not chunks:
        raise ValueError("At least one passage is needed to answer.")

    inputs = tokenizer.apply_chat_template(
        build_messages(question, chunks),
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding: the same question gives the same answer.
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "answer": text,
        "cited": cited_numbers(text, len(chunks)),
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "answer_tokens": int(new_tokens.shape[0]),
        "hit_token_limit": int(new_tokens.shape[0]) >= max_new_tokens,
    }


ENV_FILE = Path(__file__).resolve().parent / ".env"


def read_env_file(path=ENV_FILE):
    """Copy KEY=value lines from the .env file into os.environ.

    A variable that is already set in the environment wins over the file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def load_api_client():
    """Return an OpenAI client. The key comes from OPENAI_API_KEY or the .env file."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "API mode needs the openai package: .venv/bin/pip install -r requirements.txt"
        ) from error
    read_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "No OpenAI API key found. Paste it after OPENAI_API_KEY= in the .env file."
        )
    return OpenAI()


def generate_answer_api(question, chunks, client, model, max_output_tokens=MAX_API_OUTPUT_TOKENS):
    """Same prompt as generate_answer, but an OpenAI API model writes the answer."""
    if not chunks:
        raise ValueError("At least one passage is needed to answer.")

    system_message, user_message = build_messages(question, chunks)
    response = client.responses.create(
        model=model,
        instructions=system_message["content"],
        input=user_message["content"],
        max_output_tokens=max_output_tokens,
    )
    text = response.output_text.strip()
    usage = response.usage
    details = response.incomplete_details
    return {
        "answer": text,
        "cited": cited_numbers(text, len(chunks)),
        "prompt_tokens": usage.input_tokens if usage else None,
        "answer_tokens": usage.output_tokens if usage else None,
        "hit_token_limit": bool(details and details.reason == "max_output_tokens"),
    }
