"""Utility functions for the VQA BEiT‑3 project.

This module provides helper functions for reproducibility, data
pre‑processing and JSON export.  Functions here are intentionally simple
and self‑contained so that they can be used from both training and
inference scripts.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int) -> None:
    """Fix random seeds for reproducibility across Python, NumPy and PyTorch.

    Parameters
    ----------
    seed: int
        The seed value to use for all random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_mcq_json(df: pd.DataFrame, out_path: str, is_train: bool = False) -> None:
    """Convert a CSV dataframe into the BEiT‑3 multiple‑choice JSON format.

    The BEiT‑3 finetuning script expects JSON entries of the form:

    ```json
    {
      "image_id": "<image filename without extension>",
      "question": "<question text> Choices: A. <A> B. <B> C. <C> D. <D>",
      "question_id": "<identifier>",
      "answer": "<A|B|C|D>"  // optional
    }
    ```

    This function builds such a JSON list from a pandas dataframe and writes
    it to disk.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the columns ``ID``, ``img_path``, ``Question``,
        ``A``, ``B``, ``C``, ``D`` and optionally ``answer``.
    out_path: str
        Destination path for the JSON file.  Directories are created if
        they do not exist.
    is_train: bool
        Whether to include the ``answer`` field in the output.  For test
        data this should be ``False``.
    """
    mcq_data = []
    for _, row in df.iterrows():
        # Extract numeric part of the ID and construct a question_id
        num_part = re.findall(r"\d+", str(row["ID"]))
        num_part = num_part[0] if num_part else str(len(mcq_data))
        qid = f"1{num_part}1"
        # Derive image_id from the image filename (without extension)
        filename = os.path.basename(str(row["img_path"]))
        name, _ext = os.path.splitext(filename)
        # Build multiple‑choice prompt
        choices = [row["A"], row["B"], row["C"], row["D"]]
        parts = [f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)]
        mcq = f"{row['Question']} Choices: " + " ".join(parts)
        entry = {
            "image_id": name,
            "question": mcq,
            "question_id": qid,
        }
        if is_train:
            # Normalise the answer to uppercase letter
            ans = str(row.get("answer", "")).strip().upper()
            if ans not in ["A", "B", "C", "D"]:
                # Default to 'A' if the answer is missing or invalid
                ans = "A"
            entry["answer"] = ans
        mcq_data.append(entry)
    # Ensure the output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mcq_data, f, ensure_ascii=False, indent=2)