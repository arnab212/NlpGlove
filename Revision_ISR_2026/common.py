# Voice-to-Grasp - revision analyses (see README.md, section "Reproducing the paper's numbers").
# All scripts in Revision_ISR_2026 read the data folders of this repository (read-only) and write
# only to Revision_ISR_2026/outputs/. Paths are set in common.py.

"""
common.py - shared paths, labels and data loaders for the ISR revision scripts.

Nothing in here writes outside the Revision_ISR_2026 folder. All original data
is opened read-only from the parent project folder (nlpglove).
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths (resolved relative to this file, so it works on Windows and Linux)
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../nlpglove/Revision_ISR_2026
import os
ROOT = Path(os.environ.get("NLPGLOVE_ROOT", HERE.parent))   # .../nlpglove
OUT = Path(os.environ.get("REVISION_OUT", HERE / "outputs"))
OUT.mkdir(exist_ok=True)

BASE_MODEL = ROOT / "bert_finetuned_updated"       # base DistilBERT used in Exp 2-4
STUDY_DIR = ROOT / "Continual_Learning_Experiments" / "UserStudy_code_analysis"
USER_STUDY_DIR = ROOT / "user_study"
EXP1_GT = ROOT / "test_results.csv"                # 96 prompts + 4 assessor labels
DATASET_FILES = [ROOT / "dataset.json", ROOT / "updated_dataset.json"]
CL_DIR = ROOT / "Scripts_ContinualLearning"

LABELS = ["grasp", "open_hand", "gentle_grasp", "tight_grasp",
          "pinch_grasp", "tighten", "loosen"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# Experiment 4 participants: ONLY the four reported in the paper. Paper labels P1..P4 match the number of model corrections in
# Table 2 (41, 74, 86, 71). The PilotA / PilotA2 sheets in the folder are not used.
PARTICIPANTS = {
    "P1": "VoiceToGrasp_StudySheet_P1.xlsx",     # paper P1 (41 corrections)
    "P2": "VoiceToGrasp_StudySheet_P2.xlsx",     # paper P2 (74)
    "P3":  "VoiceToGrasp_StudySheet_P3.xlsx",     # paper P3 (86)
    "P4": "VoiceToGrasp_StudySheet_P4.xlsx",   # paper P4 (71)
}
PAPER_IDS = {"P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4"}


def norm_label(x) -> str | None:
    """Map free-form label strings to the 7 canonical labels."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    s = {"open": "open_hand", "openhand": "open_hand", "pinch": "pinch_grasp",
         "tight": "tight_grasp", "gentle": "gentle_grasp", "soften": "loosen"}.get(s, s)
    return s if s in LABEL2ID else None


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z ]", "", str(s).lower()).strip()


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_participant(name: str) -> pd.DataFrame:
    """Experiment 4 study sheet -> DataFrame(prompt, label). Row order kept:
    rows 0-139 adaptation phase, rows 140-209 held-out test (as in the paper)."""
    df = pd.read_excel(STUDY_DIR / PARTICIPANTS[name])
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    df = df.rename(columns={"grasp_type": "label", "true_grasp": "label"})
    df["label"] = df["label"].map(norm_label)
    df["prompt"] = df["prompt"].astype(str).str.strip()
    bad = df["label"].isna().sum()
    if bad:
        print(f"[warn] {name}: {bad} rows with unknown label dropped")
    return df.dropna(subset=["label"]).reset_index(drop=True)[["prompt", "label"]]


def load_exp1() -> pd.DataFrame:
    """96 Experiment 1 prompts. 'labels' = set of grasps chosen by the assessors
    (a prediction is correct if it is in the set - the paper's rule)."""
    rows = []
    with open(EXP1_GT, newline="", encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if not r or not r[0].strip():
                continue
            labs = {norm_label(x) for x in r[1:]} - {None}
            if labs:
                rows.append({"prompt": r[0].strip(), "labels": labs})
    return pd.DataFrame(rows)


def load_exp2() -> pd.DataFrame:
    """Experiment 2 (20 users x 10 spoken commands, Whisper transcripts).
    Accepted label set per utterance = model prediction if the user said 'yes',
    else the user's correction - pooled over both models."""
    rows = []
    for u in range(1, 21):
        b = pd.read_csv(USER_STUDY_DIR / f"bert_results_user{u}.csv")
        d = pd.read_csv(USER_STUDY_DIR / f"deepseek_results_user{u}.csv")
        for i in range(min(len(b), len(d))):
            accepted = set()
            for frame, tag in ((b, "bert"), (d, "deepseek")):
                r = frame.iloc[i]
                lab = norm_label(r["prediction"]) if str(r["feedback"]).strip().lower() == "yes" \
                    else norm_label(r["correction"])
                if lab:
                    accepted.add(lab)
            rows.append({
                "user": u, "scenario": i + 1, "prompt": str(b.iloc[i]["prompt"]).strip(),
                "labels": accepted,
                "bert_pred": norm_label(b.iloc[i]["prediction"]),
                "bert_ok": str(b.iloc[i]["feedback"]).strip().lower() == "yes",
                "deepseek_pred": norm_label(d.iloc[i]["prediction"]),
                "deepseek_ok": str(d.iloc[i]["feedback"]).strip().lower() == "yes",
            })
    return pd.DataFrame(rows)


def load_base_dataset() -> pd.DataFrame:
    """The fine-tuning set of the base DistilBERT (dataset.json + updated_dataset.json)."""
    rows = []
    for p in DATASET_FILES:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            data = [json.loads(l) for l in txt.splitlines() if l.strip()]
        for x in data:
            lab = norm_label(x.get("description")) or (
                LABELS[int(x["label"])] if str(x.get("label", "")).isdigit() else None)
            if lab:
                rows.append({"prompt": x["text"].strip(), "label": lab, "source": p.name})
    return pd.DataFrame(rows).drop_duplicates(subset=["prompt", "label"]).reset_index(drop=True)


def load_xlsx_prompts(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"true_grasp": "label", "grasp_type": "label"})
    df["label"] = df["label"].map(norm_label)
    return df.dropna(subset=["label"])[["prompt", "label"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Small stats helpers (no scipy dependency needed for these)
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
