# Voice-to-Grasp - Experiment 4 (personalisation, paper Section 2.7).
# Reads one participant's study sheet (210 written prompts: columns #, object, grasp_type, Prompt),
# adapts the base model on rows 1-140 in their order and tests the frozen model on rows 141-210.
# Writes results_training_phase.csv, results_test_phase.csv and figures to experiment_results/<user>/<time>/.
# Run : python continual_batch_learning.py   (set FILE_PATH and USER_ID below, e.g. P1)

"""
continual_batch_learning.py

Reads the VoiceToGrasp study sheet (columns: #, object, grasp_type, Prompt).
  - Rows 1-140  → continual-learning / training phase
  - Rows 141-210 → held-out test phase (model frozen)

Run:
    python continual_batch_learning.py
"""

import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from continual_trainer import ContinualTrainer, ID2LABEL

# ──────────────────────────────────────────────
# CONFIG — change these paths as needed
# ──────────────────────────────────────────────
FILE_PATH     = r"D:\PythonEnvs\nlpglove\Continual_Learning_Experiments\UserStudy_code_analysis\VoiceToGrasp_StudySheet_P1.xlsx"   # path to your XLSX
TRAIN_ROWS    = 140          # rows 1-140 (0-indexed: 0-139)
TEST_START    = 140          # rows 141-210 (0-indexed: 140-209)
USER_ID       = "P1"
RESULTS_DIR   = r"D:\PythonEnvs\nlpglove\Continual_Learning_Experiments\UserStudy_code_analysis\experiment_results"
# ──────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """Load XLSX/CSV and map columns to standard names."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Only CSV or XLSX supported.")

    # Flexible column detection
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc == "prompt":
            col_map[col] = "prompt"
        elif lc == "grasp_type" or lc == "true_grasp":
            col_map[col] = "true_grasp"

    df = df.rename(columns=col_map)

    if "prompt" not in df.columns or "true_grasp" not in df.columns:
        raise ValueError(
            f"Could not find prompt/grasp columns. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["prompt"]     = df["prompt"].astype(str).str.strip()
    df["true_grasp"] = df["true_grasp"].astype(str).str.strip()
    return df


# ──────────────────────────────────────────────
# PHASE 1: CONTINUAL LEARNING (rows 1-140)
# ──────────────────────────────────────────────

def run_training_phase(df: pd.DataFrame, trainer: ContinualTrainer) -> dict:
    """
    Evaluate-then-update on the first TRAIN_ROWS rows.
    Returns a dict with per-sample results and running accuracy.
    """
    train_df = df.iloc[:TRAIN_ROWS].reset_index(drop=True)
    total    = len(train_df)

    results   = []
    correct   = 0
    updates   = 0
    running_acc = []

    print(f"\n{'='*55}")
    print(f"  PHASE 1 — CONTINUAL LEARNING  ({total} samples)")
    print(f"{'='*55}")

    for i, row in train_df.iterrows():
        text       = row["prompt"]
        true_label = row["true_grasp"]

        pred_label, confidence = trainer.predict(text)
        is_correct = (pred_label == true_label)

        # Every sample goes into the replay buffer (correct or not)
        trainer.add_to_buffer(text, true_label)

        if is_correct:
            correct += 1
            loss_val = update_num = ""
            trainer._save_buffer()   # persist the newly added correct sample
            trainer.log_event(text, pred_label, confidence, "", "", "")
        else:
            loss_val, update_num = trainer.update_model(text, true_label)
            updates += 1
            trainer.log_event(text, pred_label, confidence, true_label, loss_val, update_num)

        running_acc.append(correct / (i + 1))

        results.append({
            "index":       i + 1,
            "prompt":      text,
            "true_label":  true_label,
            "pred_label":  pred_label,
            "confidence":  round(confidence, 4),
            "correct":     is_correct,
            "loss":        loss_val,
            "update_num":  update_num,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:>3}/{total}]  running acc={running_acc[-1]:.3f}  updates so far={updates}")

    final_acc = correct / total
    print(f"\n  ✓ Training phase complete")
    print(f"    Accuracy : {final_acc:.4f}  ({correct}/{total})")
    print(f"    Model updates : {updates}")

    return {
        "samples":     results,
        "running_acc": running_acc,
        "final_acc":   final_acc,
        "updates":     updates,
        "correct":     correct,
        "total":       total,
    }


# ──────────────────────────────────────────────
# PHASE 2: HELD-OUT TEST (rows 141-210)
# ──────────────────────────────────────────────

def run_test_phase(df: pd.DataFrame, trainer: ContinualTrainer) -> dict:
    """
    Pure inference on rows 141-210 — model is NOT updated.
    """
    test_df  = df.iloc[TEST_START:].reset_index(drop=True)
    total    = len(test_df)

    results  = []
    correct  = 0

    print(f"\n{'='*55}")
    print(f"  PHASE 2 — HELD-OUT TEST  ({total} samples, no updates)")
    print(f"{'='*55}")

    for i, row in test_df.iterrows():
        text       = row["prompt"]
        true_label = row["true_grasp"]

        pred_label, confidence = trainer.predict(text)
        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1

        results.append({
            "index":      TEST_START + i + 1,
            "prompt":     text,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": round(confidence, 4),
            "correct":    is_correct,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:>3}/{total}]  acc so far={correct/(i+1):.3f}")

    final_acc = correct / total
    print(f"\n  ✓ Test phase complete")
    print(f"    Accuracy : {final_acc:.4f}  ({correct}/{total})")

    return {
        "samples":   results,
        "final_acc": final_acc,
        "correct":   correct,
        "total":     total,
    }


# ──────────────────────────────────────────────
# ANALYSIS & VISUALISATION
# ──────────────────────────────────────────────

LABEL_ORDER = list(ID2LABEL.values())
COLORS = plt.cm.tab10.colors


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {path}")


def plot_running_accuracy(train_res: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = range(1, len(train_res["running_acc"]) + 1)
    ax.plot(xs, train_res["running_acc"], color="steelblue", linewidth=1.5)
    ax.axhline(train_res["final_acc"], color="tomato", linestyle="--",
               label=f"Final={train_res['final_acc']:.3f}")
    ax.set_xlabel("Sample #")
    ax.set_ylabel("Running Accuracy")
    ax.set_title("Phase 1 — Running Accuracy During Continual Learning")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, "01_running_accuracy.png"))


def plot_confusion_matrix(samples: list, title: str, filename: str, out_dir: str):
    y_true = [s["true_label"] for s in samples]
    y_pred = [s["pred_label"] for s in samples]

    # Only include labels that actually appear
    present = sorted(set(y_true) | set(y_pred),
                     key=lambda l: LABEL_ORDER.index(l) if l in LABEL_ORDER else 99)

    cm  = confusion_matrix(y_true, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=present)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)
    ax.set_title(title)
    _save(fig, os.path.join(out_dir, filename))


def plot_per_class_accuracy(samples: list, title: str, filename: str, out_dir: str):
    from collections import defaultdict
    by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for s in samples:
        by_class[s["true_label"]]["total"] += 1
        if s["correct"]:
            by_class[s["true_label"]]["correct"] += 1

    labels = sorted(by_class.keys(),
                    key=lambda l: LABEL_ORDER.index(l) if l in LABEL_ORDER else 99)
    accs   = [by_class[l]["correct"] / by_class[l]["total"] for l in labels]
    totals = [by_class[l]["total"] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, accs, color=[COLORS[i % 10] for i in range(len(labels))])
    for bar, t in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={t}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    _save(fig, os.path.join(out_dir, filename))


def plot_confidence_distribution(train_samples, test_samples, out_dir: str):
    t_conf = [s["confidence"] for s in train_samples]
    e_conf = [s["confidence"] for s in test_samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, conf, label, color in zip(
        axes,
        [t_conf, e_conf],
        ["Phase 1 — Training", "Phase 2 — Test"],
        ["steelblue", "darkorange"],
    ):
        ax.hist(conf, bins=20, range=(0, 1), color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(conf), color="red", linestyle="--",
                   label=f"mean={np.mean(conf):.3f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Prediction Confidence Distribution", fontweight="bold")
    _save(fig, os.path.join(out_dir, "04_confidence_distribution.png"))


def plot_update_timing(train_samples: list, out_dir: str):
    """Show which samples triggered a model update during Phase 1."""
    indices  = [s["index"] for s in train_samples]
    updated  = [1 if s["loss"] != "" else 0 for s in train_samples]

    fig, ax = plt.subplots(figsize=(12, 3))
    colors = ["tomato" if u else "steelblue" for u in updated]
    ax.bar(indices, [1] * len(indices), color=colors, width=1.0, edgecolor="none")

    update_patch = mpatches.Patch(color="tomato",    label="Model updated (wrong prediction)")
    correct_patch = mpatches.Patch(color="steelblue", label="No update (correct prediction)")
    ax.legend(handles=[update_patch, correct_patch], loc="upper right")
    ax.set_xlabel("Sample index")
    ax.set_yticks([])
    ax.set_title("Phase 1 — When Model Updates Occurred")
    ax.set_xlim(0.5, len(indices) + 0.5)
    _save(fig, os.path.join(out_dir, "05_update_timing.png"))


def plot_phase_comparison(train_res: dict, test_res: dict, out_dir: str):
    phases = ["Phase 1\nContinual Learning", "Phase 2\nHeld-out Test"]
    accs   = [train_res["final_acc"], test_res["final_acc"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(phases, accs, color=["steelblue", "darkorange"], width=0.4)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Phase Comparison — Final Accuracy")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "06_phase_comparison.png"))


def generate_analysis(train_res: dict, test_res: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print("  GENERATING ANALYSIS PLOTS")
    print(f"{'='*55}")

    plot_running_accuracy(train_res, out_dir)
    plot_confusion_matrix(train_res["samples"],
                          "Phase 1 — Confusion Matrix (Continual Learning)",
                          "02a_confusion_matrix_train.png", out_dir)
    plot_confusion_matrix(test_res["samples"],
                          "Phase 2 — Confusion Matrix (Held-out Test)",
                          "02b_confusion_matrix_test.png", out_dir)
    plot_per_class_accuracy(train_res["samples"],
                            "Phase 1 — Per-class Accuracy (Training)",
                            "03a_per_class_train.png", out_dir)
    plot_per_class_accuracy(test_res["samples"],
                            "Phase 2 — Per-class Accuracy (Test)",
                            "03b_per_class_test.png", out_dir)
    plot_confidence_distribution(train_res["samples"], test_res["samples"], out_dir)
    plot_update_timing(train_res["samples"], out_dir)
    plot_phase_comparison(train_res, test_res, out_dir)


def save_results(train_res: dict, test_res: dict, out_dir: str):
    """Save per-sample CSVs and a JSON summary."""
    pd.DataFrame(train_res["samples"]).to_csv(
        os.path.join(out_dir, "results_training_phase.csv"), index=False)
    pd.DataFrame(test_res["samples"]).to_csv(
        os.path.join(out_dir, "results_test_phase.csv"), index=False)

    train_labels = [s["true_label"] for s in train_res["samples"]]
    train_preds  = [s["pred_label"] for s in train_res["samples"]]
    test_labels  = [s["true_label"] for s in test_res["samples"]]
    test_preds   = [s["pred_label"] for s in test_res["samples"]]

    summary = {
        "experiment_timestamp": str(datetime.now()),
        "file_path": FILE_PATH,
        "user_id":   USER_ID,
        "training_phase": {
            "samples":       train_res["total"],
            "correct":       train_res["correct"],
            "accuracy":      round(train_res["final_acc"], 4),
            "model_updates": train_res["updates"],
            "classification_report": classification_report(
                train_labels, train_preds,
                labels=LABEL_ORDER, zero_division=0, output_dict=True
            ),
        },
        "test_phase": {
            "samples":  test_res["total"],
            "correct":  test_res["correct"],
            "accuracy": round(test_res["final_acc"], 4),
            "classification_report": classification_report(
                test_labels, test_preds,
                labels=LABEL_ORDER, zero_division=0, output_dict=True
            ),
        },
    }

    with open(os.path.join(out_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Pretty print to console
    print(f"\n{'='*55}")
    print("  CLASSIFICATION REPORT — TRAINING PHASE")
    print(f"{'='*55}")
    print(classification_report(train_labels, train_preds,
                                labels=LABEL_ORDER, zero_division=0))

    print(f"\n{'='*55}")
    print("  CLASSIFICATION REPORT — TEST PHASE")
    print(f"{'='*55}")
    print(classification_report(test_labels, test_preds,
                                labels=LABEL_ORDER, zero_division=0))

    print(f"\nResults saved to: {out_dir}/")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print(f"\nLoading data from: {FILE_PATH}")
    df = load_data(FILE_PATH)
    print(f"Total rows: {len(df)}  |  Training: {TRAIN_ROWS}  |  Test: {len(df) - TRAIN_ROWS}")

    trainer = ContinualTrainer(user_id=USER_ID)

    train_res = run_training_phase(df, trainer)
    test_res  = run_test_phase(df, trainer)

    out_dir = os.path.join(RESULTS_DIR, USER_ID, datetime.now().strftime("%Y%m%d_%H%M%S"))
    generate_analysis(train_res, test_res, out_dir)
    save_results(train_res, test_res, out_dir)

    print(f"\n{'='*55}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*55}")
    print(f"  Training accuracy : {train_res['final_acc']:.4f}")
    print(f"  Test accuracy     : {test_res['final_acc']:.4f}")
    print(f"  Model updates     : {train_res['updates']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
