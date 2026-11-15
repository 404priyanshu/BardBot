"""Chat with BardBot in your browser.

Usage:
    python webui.py                   # then open http://127.0.0.1:7860
    python webui.py --api-model MODEL_ID

The page runs only on this computer. The models load one time, at the start.
"""
import argparse
import html
import os

# Do not send usage data to Gradio. The page must work offline.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from bardbot import BardBot  # noqa: E402  (sets HF_HUB_OFFLINE before the model imports)
from answer import LLM_NAME, labelled_passage
from search_chunks import RERANK_CANDIDATES, chunk_location, chunk_speakers

EXAMPLES = [
    "Why does Hamlet fear what happens after death?",
    "What do the witches tell Macbeth?",
    "How does Juliet react when she learns Romeo is a Montague?",
]


def format_reply(result):
    """Return the answer and its sources as Markdown for the chat window.

    Each source is a closed <details> block: click it to read the passage.
    """
    lines = [result["answer"]]
    if result["hit_token_limit"]:
        lines.append("\n_(The answer reached the token limit and may be cut off.)_")

    lines.append("\n**Sources** (✓ = cited in the answer; click a source to read it)\n")
    for number, chunk in enumerate(result["passages"], start=1):
        marker = "✓" if number in result["cited"] else "·"
        speakers = ", ".join(chunk_speakers(chunk)) or "no speaker listed"
        location = html.escape(f"{marker} [{number}] {chunk_location(chunk)} — {speakers}")
        text = html.escape(labelled_passage(chunk)).replace("\n", "<br>")
        lines.append(f"<details><summary>{location}</summary>{text}</details>")

    lines.append(f"\n_{result['seconds']:.1f} s_")
    return "\n".join(lines)


def build_app(bot):
    import gradio as gr

    def respond(message, history):
        if not message.strip():
            return "Please ask a question."
        return format_reply(bot.ask(message))

    return gr.ChatInterface(
        respond,
        title="BardBot",
        description=(
            "Ask about *Hamlet*, *Macbeth*, or *Romeo and Juliet*. "
            f"Answers by `{bot.llm_name}`. Each question is separate: "
            "BardBot does not remember earlier questions."
        ),
        examples=EXAMPLES,
        cache_examples=False,
    )


def main():
    parser = argparse.ArgumentParser(description="BardBot chat page in the browser.")
    parser.add_argument("--top-k", type=int, default=5, help="Passages sent to the LLM.")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    parser.add_argument("--model", default=LLM_NAME, help="Local LLM in data/models.")
    parser.add_argument("--api-model", metavar="MODEL_ID",
                        help="Write answers with this OpenAI API model. Costs money.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="Open the page in the browser.")
    args = parser.parse_args()

    print("Loading models...")
    try:
        bot = BardBot(args.rerank, args.candidates, args.top_k, args.model, args.api_model)
    except RuntimeError as error:
        parser.error(str(error))
    print("Answers by:", bot.llm_name)

    # 127.0.0.1: only this computer can open the page.
    build_app(bot).launch(server_name="127.0.0.1", server_port=args.port, inbrowser=args.open)


if __name__ == "__main__":
    main()
