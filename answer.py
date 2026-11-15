"""Write an answer from retrieved passages with a local LLM."""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from search_chunks import best_device

# Hide harmless config warnings from transformers (for example about rope_parameters).
transformers_logging.set_verbosity_error()

LLM_NAME = "microsoft/Phi-4-mini-instruct"
MAX_NEW_TOKENS = 400
NO_ANSWER = "The passages do not answer this question."

SYSTEM_PROMPT = f"""You answer questions about three Shakespeare plays: Hamlet, Macbeth, and Romeo and Juliet.

The user gives you numbered passages from the plays and a question. The passages are in Shakespeare's Early Modern English. Each line starts with the name of the character who speaks it. The answer is usually in what a character says, and it is often said indirectly or in poetic language.

How to answer:
1. Find the passages that are about the topic of the question.
2. Explain the answer in your own words, in two to four sentences of plain modern English. Start directly with the explanation.
3. End every sentence with the number of the passage that supports it in square brackets, for example [2] for Passage 2.
4. You may quote one short line (at most 15 words). Never copy a whole passage or its heading.
5. Use only the passages. Do not add facts from outside knowledge.

Most questions can be answered from the passages, even when the answer is only implied by what the characters say. Give the best answer that the passages support. If they support only part of the answer, give that part and say in one sentence what the passages do not show.

Reply exactly "{NO_ANSWER}" only in these two cases:
- The question is not about Hamlet, Macbeth, or Romeo and Juliet.
- No passage has anything to do with the question.

Example:
Question: Why does Macbeth hesitate to kill the king?
Answer: Macbeth fears that the murder will bring punishment in this life and that others will copy his crime against him [1]. He also says Duncan is a good king and his guest, so he should protect him, not kill him [1]."""


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
        f"Question: {question}\nAnswer:"
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
