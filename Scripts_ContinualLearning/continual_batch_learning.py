# Voice-to-Grasp - Experiment 3 (task specialisation, paper Section 2.6): runs the continual learner
# over a prompt file (e.g. Kitchen_train_nonnative.xlsx; columns "prompts", "grasp type") and saves
# the adapted model and a log.
# Run : python continual_batch_learning.py   (set the file path in __main__)

import os
import pandas as pd
from continual_trainer import ContinualTrainer


def load_data(file_path):
    """
    Loads CSV or XLSX file.
    Must contain columns:
        - prompts
        - grasp type
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV or XLSX supported.")

    if "prompt" not in df.columns or "true_grasp" not in df.columns:
        raise ValueError("File must contain 'prompt' and 'true_grasp' columns.")

    return df


def run_continual_learning(file_path):

    df = load_data(file_path)

    trainer = ContinualTrainer(user_id="Tester-21")

    total = len(df)
    correct_predictions = 0
    update_count = 0

    print(f"Total samples: {total}")
    print("Starting evaluation...\n")

    for i, row in df.iterrows():

        text = str(row["prompt"]).strip()
        true_label = str(row["true_grasp"]).strip()

        pred_label, confidence = trainer.predict(text)

        is_correct = (pred_label == true_label)

        if is_correct:
            correct_predictions += 1

            trainer.log_event(
                input_text=text,
                predicted_label=pred_label,
                confidence=confidence,
                correct_label="",
                loss="",
                update_number=""
            )

        else:
            loss, update_number = trainer.update_model(text, true_label)
            update_count += 1

            trainer.log_event(
                input_text=text,
                predicted_label=pred_label,
                confidence=confidence,
                correct_label=true_label,
                loss=loss,
                update_number=update_number
            )

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{total}")

    final_accuracy = correct_predictions / total

    print("\n==========================")
    print("FINAL RESULTS")
    print("==========================")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Total Model Updates: {update_count}")
    print("==========================")

    return final_accuracy


if __name__ == "__main__":

    file_path = r"D:\PythonEnvs\nlpglove\Scripts_ContinualLearning\kitchen_train_nonnative.xlsx"  # Change this

    run_continual_learning(file_path)