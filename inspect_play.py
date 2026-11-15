from pathlib import Path
import xml.etree.ElementTree as ET

# Read the original XML file into a tree of elements.
path = Path("data/raw/macbeth.xml")
tree = ET.parse(path)
root = tree.getroot()

# Inspect the play.
print("Title:", root.findtext("TITLE"))
print("Number of acts:", len(root.findall("ACT")))

# Follow the structure down to the first speech.
act = root.find("ACT")
if act is None:
    raise ValueError("No ACT element found in the play.")

scene = act.find("SCENE")
if scene is None:
    raise ValueError("No SCENE element found in the first act.")

speech = scene.find("SPEECH")
if speech is None:
    raise ValueError("No SPEECH element found in the first scene.")

print("First act:", act.findtext("TITLE"))
print("First scene:", scene.findtext("TITLE"))
print("First speaker:", speech.findtext("SPEAKER"))

print("\nOpening dialogue:")
for line in speech.findall("LINE"):
    print("".join(line.itertext()))

print("\nScenes in each act:")

total_scenes = 0

for act in root.findall("ACT"):
    scenes = act.findall("SCENE")
    total_scenes += len(scenes)

    print(act.findtext("TITLE"), "—", len(scenes), "scenes")

print("Total scenes:", total_scenes)
