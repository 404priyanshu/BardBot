import json
from pathlib import Path
from statistics import median


def split_words(text, chunk_size=200, overlap=40):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be between 0 and chunk_size - 1.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunks.append({
            "text": " ".join(words[start:end]),
            "start_word": start + 1,
            "end_word": end,
        })

        if end == len(words):
            break

        start = end - overlap

    return chunks


lengths = []
scene_groups = {}

for path in sorted(Path("data/processed").glob("*.jsonl")):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)

            key = (
                record["source_file"],
                record["act"],
                record["section"],
                record["scene"],
            )

            if key not in scene_groups:
                scene_groups[key] = []

            scene_groups[key].append(record)

            text = "\n".join(
                part["text"] for part in record["parts"]
            )

            lengths.append(len(text.split()))

if not lengths:
    raise ValueError("No speech records found.")

long_speeches = [length for length in lengths if length > 300]

print("Speeches:", len(lengths))
print("Shortest speech:", min(lengths), "words")
print("Median speech:", median(lengths), "words")
print("Longest speech:", max(lengths), "words")
print("Speeches over 300 words:", len(long_speeches))

print("\nOverlap demonstration:")

demo_text = " ".join(str(number) for number in range(1, 26))

for chunk in split_words(demo_text, chunk_size=10, overlap=2):
    print(
        f"{chunk['start_word']}–{chunk['end_word']}: "
        f"{chunk['text']}"
    )

print("\nScenes and prologues:", len(scene_groups))

print("First three groups:")

for key, speeches in list(scene_groups.items())[:3]:
    source_file, act, section, scene = key
    location = scene or section.upper()

    print(
        source_file,
        act,
        location,
        "—",
        len(speeches),
        "speeches",
    )

all_chunks = []

for key, speeches in scene_groups.items():
    scene_words = []
    speech_ranges = []

    # Combine the scene while remembering which words belong to each speech.
    for speech_number, speech in enumerate(speeches, start=1):
        speech_text = "\n".join(
            f"[{part['text']}]"
            if part["kind"] == "stagedir"
            else part["text"]
            for part in speech["parts"]
        )

        words = speech_text.split()
        start_word = len(scene_words) + 1
        scene_words.extend(words)

        if words:
            speech_ranges.append({
                "speech_number": speech_number,
                "speakers": speech["speakers"],
                "start_word": start_word,
                "end_word": len(scene_words),
            })

    scene_text = " ".join(scene_words)

    for chunk_number, chunk in enumerate(
        split_words(scene_text), start=1
    ):
        speech_spans = []

        # Keep only the portion of each speech that intersects this chunk.
        for speech_range in speech_ranges:
            start = max(chunk["start_word"], speech_range["start_word"])
            end = min(chunk["end_word"], speech_range["end_word"])

            if start <= end:
                speech_spans.append({
                    "speech_number": speech_range["speech_number"],
                    "speakers": speech_range["speakers"],
                    "start_word": start,
                    "end_word": end,
                })

        # Build the model input with speaker labels, so questions that name a
        # character can match the speech. "text" stays unlabelled for display
        # and for the word ranges above.
        labelled_parts = []
        for span in speech_spans:
            span_text = " ".join(
                scene_words[span["start_word"] - 1:span["end_word"]]
            )
            label = " and ".join(span["speakers"])
            labelled_parts.append(f"{label}: {span_text}" if label else span_text)

        metadata = speeches[0]

        # Start the model input with where the chunk comes from, so questions
        # that name a play or place can match it.
        location = metadata["scene"] or metadata["section"].upper()
        header = " ".join(
            f"{metadata['title']}, {metadata['act']}, {location}".split()
        )

        chunk.update({
            "embed_text": "\n".join([header] + labelled_parts),
            "source_file": metadata["source_file"],
            "title": metadata["title"],
            "act": metadata["act"],
            "section": metadata["section"],
            "scene": metadata["scene"],
            "chunk_number": chunk_number,
            "speech_spans": speech_spans,
        })

        all_chunks.append(chunk)

print("\nTotal real chunks:", len(all_chunks))

first = all_chunks[0]
print("First chunk location:", first["title"], first["act"], first["scene"])
print("Word range:", first["start_word"], "to", first["end_word"])
print("Text preview:", first["text"][:300])
print("First speaker span:", first["speech_spans"][0])

output_folder = Path("data/chunks")
output_folder.mkdir(parents=True, exist_ok=True)
output_path = output_folder / "chunks.jsonl"

with output_path.open("w", encoding="utf-8") as output_file:
    for chunk in all_chunks:
        output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"Saved {len(all_chunks)} chunks to {output_path}")
