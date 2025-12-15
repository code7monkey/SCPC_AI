"""Entry point for model inference and submission generation.

Usage:
  python inference.py --config configs/submit.yaml

This script loads one or more trained checkpoints, runs inference on the
test set for each and then aggregates the results via majority voting.
The final answers are written to a CSV file matching the submission
format.  The script expects a YAML configuration with at least the
following fields:

```
test_csv: path to the test CSV file
tokenizer_path: path to the sentencepiece model
checkpoints: list of checkpoint file paths
output_dirs: list of directories (same length as ``checkpoints``)
sample_submission: CSV file with ``ID`` column to use as a template
output_submission: path to write the final submission CSV
```
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from collections import Counter
from typing import List

import pandas as pd

from src.trainer import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and create submission")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/submit.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def majority_vote(predictions_list: List[List[str]]) -> List[str]:
    """Compute the element‑wise majority vote across a list of predictions.

    Each element in ``predictions_list`` is a list of answer letters (``A``–``D``)
    corresponding to the same set of test questions.  The function returns
    the most common letter at each position.  If a vote is tied, the first
    occurring letter in the tie is chosen deterministically.
    """
    if not predictions_list:
        return []
    num_samples = len(predictions_list[0])
    # Ensure all prediction lists have the same length
    for preds in predictions_list:
        if len(preds) != num_samples:
            raise ValueError("All prediction lists must have the same length")
    final = []
    for i in range(num_samples):
        votes = [preds[i] for preds in predictions_list]
        most_common = Counter(votes).most_common(1)[0][0]
        final.append(most_common)
    return final


def main() -> None:
    args = parse_args()
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Read the list of checkpoints and corresponding output directories
    checkpoints: List[str] = cfg.get("checkpoints", [])
    output_dirs: List[str] = cfg.get("output_dirs", [])
    if not checkpoints:
        raise ValueError("No checkpoints specified in the configuration")
    if output_dirs and len(output_dirs) != len(checkpoints):
        raise ValueError("Length of output_dirs must match length of checkpoints")

    # Ensure test CSV exists
    test_csv = cfg.get("test_csv")
    if not test_csv or not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    tokenizer_path = cfg.get("tokenizer_path")
    sample_submission_path = cfg.get("sample_submission")
    if not sample_submission_path or not os.path.isfile(sample_submission_path):
        raise FileNotFoundError(f"Sample submission CSV not found: {sample_submission_path}")
    output_submission_path = cfg.get("output_submission", "submission.csv")

    all_predictions: List[List[str]] = []
    for i, ckpt in enumerate(checkpoints):
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
        print(f"Running inference for checkpoint {ckpt}")
        # Build a config for this inference round
        infer_cfg = {
            "test_csv": test_csv,
            "tokenizer_path": tokenizer_path,
            "model_ckpt": ckpt,
            "num_classes": int(cfg.get("num_classes", 4)),
            "batch_size": int(cfg.get("batch_size", 32)),
            "device": cfg.get("device", None),
        }
        preds = run_inference(infer_cfg)
        all_predictions.append(preds)

    # If only one set of predictions, no need for majority vote
    if len(all_predictions) == 1:
        final_preds = all_predictions[0]
    else:
        final_preds = majority_vote(all_predictions)

    # Write submission CSV
    sample = pd.read_csv(sample_submission_path)
    if "answer" not in sample.columns:
        # If the sample submission does not contain an answer column, append one
        sample["answer"] = final_preds
    else:
        sample["answer"] = final_preds
    out_dir = os.path.dirname(output_submission_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    sample.to_csv(output_submission_path, index=False)
    print(f"Submission file written to {output_submission_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)