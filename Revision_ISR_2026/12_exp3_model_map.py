# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
12_exp3_model_map.py - which adapted model and which test files produce the Experiment 3
numbers in the paper (kitchen 0.72 -> 0.90, repair 0.82 -> 0.77)?

For every candidate continual-learning run (Tester-8, -9, -10, -20, -21) it reports
  * which prompt file(s) the run adapted on (by matching the prompts in its log.csv)
  * the accuracy of the base model and of the adapted model on each of the four test files.
Read-only; writes outputs/exp3_model_map.md.   Run:  python 12_exp3_model_map.py
"""
import re

import numpy as np
import pandas as pd

from cl_core import load_base, predict
from common import BASE_MODEL, CL_DIR, LABELS, OUT, ROOT, load_xlsx_prompts

n = lambda t: re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()
FILES = ["Kitchen_train.xlsx", "Kitchen_train_nonnative.xlsx", "kitchen_test.xlsx",
         "kitchen_test_nonnative.xlsx", "repair_test.xlsx", "repair_test_nonnative.xlsx"]
TESTS = ["kitchen_test.xlsx", "kitchen_test_nonnative.xlsx", "repair_test.xlsx", "repair_test_nonnative.xlsx"]
data = {f: load_xlsx_prompts(CL_DIR / f) for f in FILES}
RUNS = ["Tester-8", "Tester-9", "Tester-10", "Tester-20", "Tester-21"]
out = ["# Experiment 3: which run and which files give the paper's numbers\n",
       "Paper: kitchen 0.72 -> 0.90; repair 0.82 -> 0.77\n"]


def acc(model_dir, T):
    tok, m = load_base(model_dir)
    ids, _ = predict(m, tok, list(T.prompt))
    return float(np.mean([LABELS[i] == l for i, l in zip(ids, T.label)]))


out.append("## Base model (bert_finetuned_updated)\n")
out.append("| " + " | ".join(TESTS) + " |\n|" + "---|" * len(TESTS))
out.append("| " + " | ".join(f"{acc(BASE_MODEL, data[t]):.3f}" for t in TESTS) + " |\n")

out.append("## Adapted runs\n")
out.append("| run | prompts in log | updates | adapted on (prompt overlap with each file) | " + " | ".join(TESTS) + " |")
out.append("|---|---|---|---|" + "---|" * len(TESTS))
for r in RUNS:
    d = ROOT / "Continual_Learning_Experiments" / r
    log = pd.read_csv(d / "log.csv", encoding="cp1252")
    seen = set(map(n, log.input_text))
    ov = ", ".join(f"{f.replace('.xlsx', '')}: {len(seen & set(map(n, data[f].prompt)))}/{len(data[f])}"
                   for f in FILES if len(seen & set(map(n, data[f].prompt))))
    upd = int(log.update_number.notna().sum())
    accs = " | ".join(f"{acc(d / 'adaptive_model', data[t]):.3f}" for t in TESTS)
    out.append(f"| {r} | {len(log)} | {upd} | {ov or 'none of the current files'} | {accs} |")

(OUT / "exp3_model_map.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
