import json
from pathlib import Path
import xml.etree.ElementTree as ET


def read_text(element):
    text = element.text or ""

    for child in element:
        if child.tag != "STAGEDIR":
            raise ValueError(f"Unexpected inline element: {child.tag}")

        text += f" [{read_text(child)}] "
        text += child.tail or ""

    return " ".join(text.split())

source_folder = Path("data/raw")
files = sorted(source_folder.glob("*.xml"))

if not files:
    raise ValueError("No XML files found in data/raw.")

for path in files:
    root = ET.parse(path).getroot()

    if root.tag != "PLAY":
        raise ValueError(f"Expected a PLAY element in {path.name}")

    title = root.findtext("TITLE")
    acts = root.findall("ACT")
    scenes = root.findall("ACT/SCENE")
    prologues = root.findall("ACT/PROLOGUE")

    print(f"\nFile: {path.name}")
    print(f"Title: {title}")
    print(f"Acts: {len(acts)}")
    print(f"Scenes: {len(scenes)}")
    print(f"Prologues: {len(prologues)}")

    records = []

    for act in root.findall("ACT"):
        for section in act:
            if section.tag not in ("SCENE", "PROLOGUE"):
                continue

            for speech in section.findall("SPEECH"):
                speakers = []

                for speaker in speech.findall("SPEAKER"):
                    name = (speaker.text or "").strip()

                    if name:
                        speakers.append(name)

                parts = []

                for child in speech:
                    if child.tag not in ("LINE", "STAGEDIR"):
                        continue

                    parts.append({
                        "kind": child.tag.lower(),
                        "text": read_text(child),
                    })

                record = {
                    "source_file": path.name,
                    "title": title,
                    "act": act.findtext("TITLE"),
                    "section": section.tag.lower(),
                    "scene": (
                        section.findtext("TITLE")
                        if section.tag == "SCENE"
                        else None
                    ),
                    "speakers": speakers,
                    "parts": parts,
                }

                records.append(record)

    print(f"Speech records: {len(records)}")
    print("First record:")
    print(json.dumps(records[0], indent=2))

    output_path = Path("data/processed") / f"{path.stem}.jsonl"

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            line = json.dumps(record, ensure_ascii=False)
            output_file.write(line + "\n")

    print(f"Saved {len(records)} speech records to {output_path}")
