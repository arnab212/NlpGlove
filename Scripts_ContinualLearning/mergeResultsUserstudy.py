# Voice-to-Grasp - merges the Experiment 2 result files (user_study/*.csv) into one file of prompts
# with the participant-confirmed or corrected grasp (merged_corrected_prompts.csv).
# Set DATA_FOLDER and OUTPUT_FILE below.

import os
import pandas as pd
from glob import glob

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

DATA_FOLDER = r"D:\PythonEnvs\nlpglove\user_study"   # Change if files are elsewhere
OUTPUT_FILE = r"D:\PythonEnvs\nlpglove\Scripts_ContinualLearning\merged_corrected_prompts.csv"

# Your exact column names
PROMPT_COL = "prompt"
PRED_COL = "prediction"
FEEDBACK_COL = "feedback"
CORRECTION_COL = "correction"

# -------------------------------------------------------
# Decide final grasp type
# -------------------------------------------------------

def get_final_grasp(row):
    """
    If feedback says incorrect, use correction.
    Otherwise use prediction.
    """
    feedback_value = str(row[FEEDBACK_COL]).strip().lower()

    if feedback_value in ["incorrect", "wrong", "no", "0"]:
        return row[CORRECTION_COL]
    else:
        return row[PRED_COL]


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    file_pattern = os.path.join(DATA_FOLDER, "bert_results_user*.csv")
    files = sorted(glob(file_pattern))

    if not files:
        print("No bert_results_user*.csv files found.")
        return

    print(f"Found {len(files)} files.")

    all_data = []

    for file in files:
        print(f"Reading: {file}")
        df = pd.read_csv(file)

        # Validate required columns
        required_cols = [PROMPT_COL, PRED_COL, FEEDBACK_COL, CORRECTION_COL]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {file}")

        # Compute final grasp type
        df["final_grasp"] = df.apply(get_final_grasp, axis=1)

        # Keep only required columns
        df_final = df[[PROMPT_COL, "final_grasp"]].copy()
        df_final.columns = ["prompts", "grasp type"]

        all_data.append(df_final)

    # Merge all users
    merged_df = pd.concat(all_data, ignore_index=True)

    # Optional: remove exact duplicates
    merged_df = merged_df.drop_duplicates()

    # Save output
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved merged dataset to: {OUTPUT_FILE}")
    print(f"Total samples: {len(merged_df)}")


if __name__ == "__main__":
    main()