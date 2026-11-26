# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""Quick check: do the prompts used in the learning-rate search (Tester-1..6 logs, 194 prompts each)
overlap with the Experiment 3 adaptation / test prompts?  Read-only: no data file is modified.

Run from anywhere:  python 09_check_search_overlap.py
"""
import difflib
import os
import re

import pandas as pd

ROOT = os.environ.get("NLPGLOVE_ROOT", r"D:\PythonEnvs\nlpglove")
CL = os.path.join(ROOT, "Scripts_ContinualLearning")
LOGS = os.path.join(ROOT, "Continual_Learning_Experiments")


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def load_xlsx(name):
    df = pd.read_excel(os.path.join(CL, name))
    return [norm(p) for p in df["prompt"]]


sets = {name: load_xlsx(name) for name in [
    "Kitchen_train.xlsx", "Kitchen_train_nonnative.xlsx",
    "kitchen_test.xlsx", "kitchen_test_nonnative.xlsx",
    "repair_test.xlsx", "repair_test_nonnative.xlsx"]}

search = {}
for t in range(1, 7):
    f = os.path.join(LOGS, f"Tester-{t}", "log.csv")
    search[f"Tester-{t}"] = [norm(p) for p in pd.read_csv(f, encoding="cp1252")["input_text"]]

ref = search["Tester-1"]
print("Search runs use the same prompts:",
      all(sorted(v) == sorted(ref) for v in search.values()))
print(f"Search prompts: {len(ref)} ({len(set(ref))} unique)\n")

print(f"{'Experiment 3 file':32s} {'n':>4s} {'exact in search':>16s} {'near-dup (>=0.85)':>18s}")
for name, prompts in sets.items():
    exact = sum(p in set(ref) for p in prompts)
    near = sum(1 for p in prompts if p not in set(ref)
               and max(difflib.SequenceMatcher(None, p, q).ratio() for q in ref) >= 0.85)
    print(f"{name:32s} {len(prompts):4d} {exact:16d} {near:18d}")

# which file(s) make up the 194 search prompts?
union = set().union(*sets.values())
print(f"\nSearch prompts found in any Experiment 3 file: {sum(p in union for p in set(ref))} of {len(set(ref))}")
