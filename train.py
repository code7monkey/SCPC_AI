"""Entry point for model training.

Usage:
  python train.py --config configs/train.yaml

This script reads a YAML configuration file, splits the training CSV
into train/validation sets (if necessary), exports JSON files for the
BEiT‑3 finetuning script and then invokes the high‑level training
function in ``src.trainer``.  The splitting and JSON export are
performed here to avoid recomputing them on every call to
``run_training``.
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml

import pandas as pd

from src.dataset import load_dataframes, split_dataset
from src.utils import make_mcq_json
from src.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BEiT‑3 VQA model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load and split the training dataframe if the validation CSV does not exist
    train_csv_path = cfg.get("train_csv")
    val_csv_path = cfg.get("val_csv")
    if not train_csv_path or not os.path.isfile(train_csv_path):
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    if not val_csv_path:
        # Perform a train/val split and write to temporary CSVs
        train_split, val_split = split_dataset(
            train_df,
            test_size=float(cfg.get("val_split", 0.15)),
            random_state=int(cfg.get("seed", 42)),
        )
        val_csv_path = os.path.join(os.path.dirname(train_csv_path), "val_split.csv")
        train_csv_path = os.path.join(os.path.dirname(train_csv_path), "train_split.csv")
        train_split.to_csv(train_csv_path, index=False)
        val_split.to_csv(val_csv_path, index=False)
        cfg["train_csv"] = train_csv_path
        cfg["val_csv"] = val_csv_path
    else:
        # Use provided validation CSV
        if not os.path.isfile(val_csv_path):
            raise FileNotFoundError(f"Validation CSV not found: {val_csv_path}")
        cfg["train_csv"] = train_csv_path
        cfg["val_csv"] = val_csv_path

    # Export JSON files required by BEiT‑3 finetuning script
    json_dir = cfg.get("json_dir", "json_file")
    os.makedirs(json_dir, exist_ok=True)
    train_json = os.path.join(json_dir, "train.json")
    val_json = os.path.join(json_dir, "val.json")
    test_json = os.path.join(json_dir, "test.json")
    # Generate JSON from CSVs
    make_mcq_json(pd.read_csv(cfg["train_csv"]), train_json, is_train=True)
    make_mcq_json(pd.read_csv(cfg["val_csv"]), val_json, is_train=True)
    if "test_csv" in cfg and os.path.isfile(cfg["test_csv"]):
        make_mcq_json(pd.read_csv(cfg["test_csv"]), test_json, is_train=False)
        cfg["test_json"] = test_json

    # Ensure the output directory exists
    output_dir = cfg.get("output_dir", "finetuned_model_file")
    os.makedirs(output_dir, exist_ok=True)
    cfg["output_dir"] = output_dir

    # Kick off training
    run_training(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)