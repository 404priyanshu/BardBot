import json
from pathlib import Path
import xml.etree.ElementTree as ET


def check(condition, message):
    if not condition:
        raise ValueError(message)


def plain_text(text):
    # Ignore added stage brackets and differences in whitespace.
    text = text.replace("[", "").replace("]", "")
    return " ".join(text.split())


for source_path in sorted(Path("data/raw").glob("*.xml")):
    root = ET.parse(source_path).getroot()
    speeches = root.findall(".//SPEECH")

    expected_locations = []

    for act in root.findall("ACT"):
        for section in act:
            if section.tag not in ("SCENE", "PROLOGUE"):
                continue

            for speech in section.findall("SPEECH"):
                expected_locations.append({
                    "source_file": source_path.name,
                    "act": act.findtext("TITLE"),
                    "section": section.tag.lower(),
                    "scene": (
                        section.findtext("TITLE")
                        if section.tag == "SCENE"
                        else None
                    ),
                })

    check(
        len(expected_locations) == len(speeches),
        f"{source_path.name}: a speech has no recognized location",
    )

    output_path = Path("data/processed") / f"{source_path.stem}.jsonl"

    with output_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]

    check(
        len(records) == len(speeches),
        f"{source_path.name}: speech count mismatch",
    )

    for number, (record, speech) in enumerate(
        zip(records, speeches), start=1
    ):
        label = f"{source_path.name}, speech {number}"

        location = expected_locations[number - 1]

        for field, expected in location.items():
            check(
                record[field] == expected,
                f"{label}: incorrect {field}",
            )

        check(
            record["title"] == root.findtext("TITLE"),
            f"{label}: incorrect play title",
        )

        speakers = []
        for speaker in speech.findall("SPEAKER"):
            name = (speaker.text or "").strip()
            if name:
                speakers.append(name)

        check(record["speakers"] == speakers, f"{label}: wrong speakers")

        original_parts = [
            child for child in speech
            if child.tag in ("LINE", "STAGEDIR")
        ]

        check(
            len(record["parts"]) == len(original_parts),
            f"{label}: missing or extra speech parts",
        )

        for part, original in zip(record["parts"], original_parts):
            check(
                part["kind"] == original.tag.lower(),
                f"{label}: incorrect part kind or order",
            )

            original_text = " ".join(original.itertext())

            check(
                plain_text(part["text"]) == plain_text(original_text),
                f"{label}: text differs from the original",
            )

    print(f"PASS: {output_path.name} — {len(records)} speeches checked")
