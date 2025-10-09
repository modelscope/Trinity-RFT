"""
This script processes a single model's score directory to calculate and summarize performance metrics.

It reads all 'BFCL_v3_*_score.json' files within a specified directory, calculates
weighted average accuracies ('by_instance'), and saves the results as a single-row
markdown table in a .txt file.

The target directory is provided as a command-line argument.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd


def add_average_columns(df: pd.DataFrame) -> None:
    """
    Calculates and adds weighted average accuracy columns to the DataFrame in-place.

    This function computes:
    - 'live_average_by_instance': Weighted average for 'live' scores.
    - 'nonlive_average_by_instance': Weighted average for 'non-live' scores.
    - 'overall_average_by_instance': Overall weighted average.

    Args:
        df (pd.DataFrame): A single-row DataFrame containing the model's scores.
    """
    # Find all 'live' and 'non-live' accuracy columns
    live_accuracy_cols = [c for c in df.columns if "live" in c and c.endswith("_accuracy")]
    nonlive_accuracy_cols = [c for c in df.columns if "live" not in c and c.endswith("_accuracy")]

    # Get corresponding count columns to calculate weighted averages
    live_correct_cols = [c.replace("_accuracy", "_correct_count") for c in live_accuracy_cols]
    live_total_cols = [c.replace("_accuracy", "_total_count") for c in live_accuracy_cols]
    nonlive_correct_cols = [c.replace("_accuracy", "_correct_count") for c in nonlive_accuracy_cols]
    nonlive_total_cols = [c.replace("_accuracy", "_total_count") for c in nonlive_accuracy_cols]

    # --- Calculate and Add Weighted Average Columns ---
    live_correct_sum = df[live_correct_cols].sum(axis=1) if live_correct_cols else 0
    live_total_sum = df[live_total_cols].sum(axis=1) if live_total_cols else 0
    nonlive_correct_sum = df[nonlive_correct_cols].sum(axis=1) if nonlive_correct_cols else 0
    nonlive_total_sum = df[nonlive_total_cols].sum(axis=1) if nonlive_total_cols else 0

    # Calculate weighted averages, checking for division by zero
    if (
        isinstance(live_total_sum, pd.Series)
        and not live_total_sum.empty
        and live_total_sum.iloc[0] > 0
    ):
        df["live_average_by_instance"] = live_correct_sum / live_total_sum
    else:
        df["live_average_by_instance"] = 0.0

    if (
        isinstance(nonlive_total_sum, pd.Series)
        and not nonlive_total_sum.empty
        and nonlive_total_sum.iloc[0] > 0
    ):
        df["nonlive_average_by_instance"] = nonlive_correct_sum / nonlive_total_sum
    else:
        df["nonlive_average_by_instance"] = 0.0

    # Calculate the overall weighted average
    overall_correct_sum = live_correct_sum + nonlive_correct_sum
    overall_total_sum = live_total_sum + nonlive_total_sum
    if (
        isinstance(overall_total_sum, pd.Series)
        and not overall_total_sum.empty
        and overall_total_sum.iloc[0] > 0
    ):
        df["overall_average_by_instance"] = overall_correct_sum / overall_total_sum
    else:
        df["overall_average_by_instance"] = 0.0


# --- Main execution block ---
if __name__ == "__main__":
    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Process a single model's score directory and generate a summary report."
    )
    parser.add_argument(
        "target_dir",
        type=str,
        help="The full path to the single model directory you want to process.",
    )
    args = parser.parse_args()

    target_directory = args.target_dir

    # --- Start Processing ---
    if not os.path.isdir(target_directory):
        print(f"Error: Directory not found -> {target_directory}")
        sys.exit(1)  # Exit with an error code

    # 1. The model name is simply the name of the target directory.
    model_name = os.path.basename(os.path.normpath(target_directory))

    # 2. Read score data from all relevant JSON files in the directory.
    flat_data = {}
    score_files = glob.glob(os.path.join(target_directory, "BFCL_v3_*_score.json"))

    if not score_files:
        print(f"Warning: No '*_score.json' files found in {target_directory}. Exiting.")
        sys.exit(0)

    for score_file in score_files:
        score_type = os.path.basename(score_file).replace("BFCL_v3_", "").replace("_score.json", "")
        try:
            with open(score_file, "r", encoding="utf8") as f:
                data = json.loads(f.readline())
                # Flatten the data directly into the dictionary
                flat_data[f"{score_type}_accuracy"] = data.get("accuracy", 0)
                flat_data[f"{score_type}_correct_count"] = data.get("correct_count", 0)
                flat_data[f"{score_type}_total_count"] = data.get("total_count", 0)
        except Exception as e:
            print(f"Error reading file {score_file}: {e}")

    # 3. Create a single-row DataFrame for the model.
    df = pd.DataFrame([flat_data], index=[model_name])

    # 4. Call the function to calculate and add the average columns.
    add_average_columns(df)

    # 5. Define output file path and save the result.
    output_path = f"result_{model_name}.txt"
    pd.set_option("display.width", 2000)  # Ensure the markdown table doesn't wrap lines.

    try:
        with open(output_path, "w", encoding="utf8") as f:
            f.write(df.to_markdown())
        print(f"Successfully processed '{model_name}'.")
        print(f"Result saved to: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")
