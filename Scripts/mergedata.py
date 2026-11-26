# Voice-to-Grasp - fine-tuning data: extracts the JSON records from dataset.txt and writes them as one
# JSON dataset (paper Section 2.3).

import json
import re

# Read the raw content from the input file
with open("dataset.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

# Use regex to extract all valid JSON objects
json_objects = re.findall(r'\{.*?\}', raw_data)

parsed_data = []
for obj in json_objects:
    try:
        parsed_data.append(json.loads(obj))
    except json.JSONDecodeError as e:
        print(f"Error parsing object: {obj}")
        print(f"Error: {e}")

# Option 1: Save as line-delimited JSON (for HuggingFace/transformers, etc.)
with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for item in parsed_data:
        f.write(json.dumps(item) + "\n")

# Option 2: Save as a regular JSON array (optional)
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(parsed_data, f, indent=2)

print("✅ Conversion complete!")
