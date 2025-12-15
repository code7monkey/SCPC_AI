"""Dataset utilities for the VQA BEiT‑3 project.

This module defines data loading helpers for both training and inference.
It includes a PyTorch ``Dataset`` implementation tailored to the Visual7W
multiple‑choice format.  Each item in the dataset yields tokenised inputs
compatible with the ``VQAModel`` defined in ``model.py``.  Images are
currently ignored – BEiT‑3 uses both text and vision inputs, but for
simplicity this example focuses on the textual component.  You can extend
the ``__getitem__`` method to load and process image pixels using
``PIL.Image`` and torchvision transforms if you wish.
"""

from __future__ import annotations

import os
from typing import Tuple, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer


class Visual7WDataset(Dataset):
    """PyTorch dataset for Visual7W multiple‑choice QA.

    Parameters
    ----------
    csv_path: str
        Path to the CSV file containing the formatted Visual7W data.  The
        CSV must include columns ``ID``, ``img_path``, ``Question``, ``A``,
        ``B``, ``C``, ``D`` and optionally ``answer``.
    tokenizer_path: str
        Path to a sentencepiece or tokenizer folder.  See ``model.VQAModel``
        for details.  The tokenizer is loaded once per dataset.
    image_root: Optional[str]
        Directory containing the image files.  This argument is reserved
        for future use when vision features are incorporated.
    is_train: bool
        Indicates whether the dataset should yield labels.  If ``True``, the
        ``answer`` column is converted into an integer label in the range
        ``0``–``num_classes-1``.  If ``False``, labels are omitted from
        the returned tuple.
    max_length: int
        Maximum sequence length for tokenisation.  Longer sequences will be
        truncated and shorter ones padded.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_path: str,
        image_root: Optional[str] = None,
        is_train: bool = True,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
        self.df = pd.read_csv(csv_path)
        if is_train and "answer" not in self.df.columns:
            raise ValueError("Training data must contain an 'answer' column")
        self.is_train = is_train
        self.image_root = image_root
        # Load tokenizer once and reuse
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
            except Exception:
                self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        else:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def _build_text(self, row: pd.Series) -> str:
        """Concatenate question and choices into a single prompt.

        The BEiT‑3 finetuning script expects the question followed by the
        multiple‑choice options.  This helper assembles the string in the
        appropriate format: ``"<Q> Choices: A. <A> B. <B> C. <C> D. <D>"``.
        """
        choices = [row["A"], row["B"], row["C"], row["D"]]
        parts = [f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)]
        text = f"{row['Question']} Choices: " + " ".join(parts)
        return text

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = self._build_text(row)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        if self.is_train:
            label_char = str(row["answer"]).strip().upper()
            if label_char not in ["A", "B", "C", "D"]:
                raise ValueError(f"Invalid answer label '{label_char}' in row {idx}")
            label = ord(label_char) - ord("A")
            return input_ids, attention_mask, label
        else:
            return input_ids, attention_mask


def load_dataframes(train_csv: str, test_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data from CSV files.

    Parameters
    ----------
    train_csv: str
        Path to the training CSV file.
    test_csv: str
        Path to the test CSV file.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        A tuple containing the loaded training and test dataframes.
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, test_df


def split_dataset(
    train_df: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training dataframe into train and validation partitions.

    Returns a pair of dataframes: ``(train_split, val_split)``.  Uses
    scikit‑learn's ``train_test_split`` under the hood.
    """
    return train_test_split(train_df, test_size=test_size, random_state=random_state)