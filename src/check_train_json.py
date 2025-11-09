import json
from collections import Counter
import pprint

with open("data/train.json", "r") as f:
    data = json.load(f)

labels = [e["label"] for e in data]
print("Total entries:", len(data))
print("Label counts:", Counter(labels))
print("\nFirst 5 entries:")
pprint.pprint(data[:5])
