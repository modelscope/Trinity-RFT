import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing import event_accumulator

from trinity.utils.log import get_logger

# Initialize logger
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot multi results from TensorBoard logs.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}", exc_info=True)
            raise ValueError(f"Error parsing YAML file: {e}")
    return config


def find_scalars_in_event_file(event_file: str) -> list[str]:
    """Scans a single tfevents file and returns all scalar keys"""
    try:
        ea = event_accumulator.EventAccumulator(
            event_file, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        return ea.scalars.Keys()
    except Exception as e:
        logger.warning(f"Could not read scalars from event file '{event_file}': {e}")
        return []


def build_scalar_location_map(base_path: str) -> dict[str, str]:
    """Find all scalars in 'explorer' and 'trainer'"""
    scalar_map = {}
    for folder in ["explorer", "trainer"]:
        log_dir = os.path.join(base_path, "monitor", "tensorboard", folder)
        if not os.path.isdir(log_dir):
            continue

        event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        if not event_files:
            continue

        # Use the first event file found in the directory
        keys = find_scalars_in_event_file(event_files[0])
        for key in keys:
            if key in scalar_map:
                logger.warning(
                    f"Duplicate scalar key '{key}' found. Using first one found ('{scalar_map[key]}')."
                )
            else:
                scalar_map[key] = folder
    return scalar_map


def find_tfevents_file(dir_path: str) -> str | None:
    """Finds a tfevents file within a specified directory"""
    event_files = glob.glob(os.path.join(dir_path, "events.out.tfevents.*"))
    if not event_files:
        return None
    if len(event_files) > 1:
        latest_file = sorted(event_files)[-1]
        logger.debug(
            f"Multiple tfevents files found in '{dir_path}'. Using the latest one: {latest_file}"
        )
        return latest_file
    return event_files[0]


def parse_tensorboard_log(log_dir: str, scalar_key: str) -> pd.Series:
    """Parses a single TensorBoard log directory to extract scalar data"""
    try:
        event_file = find_tfevents_file(log_dir)
        if event_file is None:
            raise FileNotFoundError(f"No tfevents file found in directory: '{log_dir}'")

        ea = event_accumulator.EventAccumulator(
            event_file, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        if scalar_key not in ea.scalars.Keys():
            logger.warning(f"Scalar key '{scalar_key}' not found in file '{event_file}'.")
            return pd.Series(dtype=np.float64)

        scalar_events = ea.scalars.Items(scalar_key)
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]

        return pd.Series(data=values, index=steps, name=log_dir)

    except Exception as e:
        logger.error(f"Failed to parse directory '{log_dir}': {e}")
        return pd.Series(dtype=np.float64)


def plot_confidence_interval(
    experiments_data: dict, title: str, x_label: str, y_label: str, output_filename: str
):
    """Plots the mean and confidence interval for multiple experiments"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (exp_name, exp_details) in enumerate(experiments_data.items()):
        all_runs_data = exp_details["data"]
        color = exp_details.get("color") or color_cycle[i % len(color_cycle)]

        if not all_runs_data:
            logger.warning(f"No valid data for experiment '{exp_name}' on this plot. Skipping.")
            continue

        df = pd.concat(all_runs_data, axis=1)
        mean_values = df.mean(axis=1).sort_index()
        std_values = df.std(axis=1).sort_index()
        steps = mean_values.index.values

        ax.plot(
            steps, mean_values, label=exp_name, color=color, marker="o", markersize=4, linestyle="-"
        )
        ax.fill_between(
            steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2
        )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc="best", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)

    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    logger.info(f"Chart successfully saved to '{output_filename}'")
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Successfully loaded configuration from: {args.config}")

    # Extract settings
    plot_cfg = config.get("plot_configs", {})
    exps_cfg = config.get("exps_configs", {})

    output_path = plot_cfg.get("output_path", "./plots")
    scalar_keys_to_plot = plot_cfg.get("scalar_keys", [])

    if not scalar_keys_to_plot:
        logger.warning("No 'scalar_keys' specified in 'plot_configs'.")
        return

    # Build scalar location maps for each experiment group
    scalar_maps = {}
    for exp_name, exp_details in exps_cfg.items():
        logger.info(f"Scanning for scalars in experiment group: {exp_name}")
        for path in exp_details.get("paths", []):
            if os.path.isdir(path):
                scalar_maps[exp_name] = build_scalar_location_map(path)
                if scalar_maps[exp_name]:
                    logger.info(
                        f"Scalar map for '{exp_name}' created successfully from path: {path}"
                    )
                    break
        if exp_name not in scalar_maps:
            logger.warning(
                f"Could not create a scalar map for '{exp_name}'. All paths might be invalid."
            )
            scalar_maps[exp_name] = {}

    # Main Loop: Generate one plot for each specified scalar key
    for scalar_key in scalar_keys_to_plot:
        logger.info(f"\n--- Generating plot for scalar_key: '{scalar_key}' ---")
        experiments_data_for_this_plot = {}

        for exp_name, exp_details in exps_cfg.items():
            scalar_map = scalar_maps.get(exp_name, {})
            if scalar_key not in scalar_map:
                logger.warning(
                    f"Scalar '{scalar_key}' not found for experiment '{exp_name}'. Skipping this curve."
                )
                continue

            target_folder = scalar_map[scalar_key]
            logger.info(
                f"Processing '{exp_name}': Found '{scalar_key}' in '{target_folder}' folder."
            )

            all_runs_data = []
            for path in exp_details.get("paths", []):
                log_dir = os.path.join(path, "monitor", "tensorboard", target_folder)
                if os.path.isdir(log_dir):
                    run_data = parse_tensorboard_log(log_dir, scalar_key)
                    if not run_data.empty:
                        all_runs_data.append(run_data)
                else:
                    logger.warning(f"Log directory not found for path: {log_dir}")

            experiments_data_for_this_plot[exp_name] = {
                "data": all_runs_data,
                "color": exp_details.get("color"),
            }

        # Generate dynamic plot details
        clean_scalar_name = re.sub(r"[^a-zA-Z0-9_-]", "_", scalar_key)
        output_filename = os.path.join(output_path, f"{clean_scalar_name}.png")

        # Use templates for titles and labels if available
        title = plot_cfg.get("title", "{scalar_key}").format(scalar_key=scalar_key)
        x_label = plot_cfg.get("x_label", "Step")
        y_label = plot_cfg.get("y_label_template", "{scalar_key}").format(scalar_key=scalar_key)

        plot_confidence_interval(
            experiments_data=experiments_data_for_this_plot,
            title=title,
            x_label=x_label,
            y_label=y_label,
            output_filename=output_filename,
        )
    logger.info("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
