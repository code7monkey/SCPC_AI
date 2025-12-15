"""Training and inference utilities for the VQA BEiT‑3 project.

This module encapsulates the training loop and inference routines.  It
provides two high‑level functions, ``run_training`` and ``run_inference``,
which take dictionaries of configuration parameters (typically loaded
from YAML files) and perform the appropriate tasks.

The implementation here uses a simple PyTorch training loop for
illustration.  If you wish to delegate training to the original
``run_beit3_finetuning.py`` script provided by Microsoft, you can do so
by invoking it with :func:`subprocess.run` from within these functions.
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import VQAModel
from .dataset import Visual7WDataset
from .losses import CrossEntropyLoss
from .utils import seed_everything


def _save_checkpoint(model: VQAModel, path: str) -> None:
    """Save the model's state dict to the given path, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def run_training(cfg: Dict) -> None:
    """Train a VQA model based on parameters supplied via a config dictionary.

    The config should contain at least the following keys:

    ``train_csv``: Path to the training CSV file.
    ``val_csv``: Path to the validation CSV file.
    ``tokenizer_path``: Path to the sentencepiece model or tokenizer.
    ``model_ckpt``: Optional path to a pre‑existing model checkpoint to
        initialise from.
    ``output_dir``: Directory in which to save checkpoints.
    ``num_classes``: Number of answer classes.
    ``batch_size``: Mini‑batch size.
    ``epochs``: Number of training epochs.
    ``learning_rate``: Learning rate for the optimiser.
    ``seed``: Random seed.

    Additional keys are ignored.  The function prints basic training
    progress to stdout.
    """
    # Fix seeds and determine device
    seed_everything(int(cfg.get("seed", 42)))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = VQAModel(
        model_ckpt_path=cfg.get("model_ckpt"),
        num_classes=int(cfg.get("num_classes", 4)),
        tokenizer_path=cfg.get("tokenizer_path"),
        device=device,
    )
    model.to(device)

    # Prepare datasets and loaders
    train_dataset = Visual7WDataset(
        csv_path=cfg["train_csv"],
        tokenizer_path=cfg.get("tokenizer_path"),
        is_train=True,
    )
    val_dataset = Visual7WDataset(
        csv_path=cfg["val_csv"],
        tokenizer_path=cfg.get("tokenizer_path"),
        is_train=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=int(cfg.get("batch_size", 16)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg.get("batch_size", 16)), shuffle=False)

    # Loss and optimiser
    criterion = CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 2e-5)))

    num_epochs = int(cfg.get("epochs", 1))
    best_val_acc = 0.0
    os.makedirs(cfg.get("output_dir", "checkpoints"), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimiser.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * input_ids.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # Simple validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")
        # Save checkpoint after each epoch
        ckpt_path = os.path.join(cfg.get("output_dir", "checkpoints"), f"checkpoint-{epoch}.pth")
        _save_checkpoint(model, ckpt_path)
        # Track the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(cfg.get("output_dir", "checkpoints"), "checkpoint-best.pth")
            _save_checkpoint(model, best_path)


def run_inference(cfg: Dict) -> List[str]:
    """Generate predictions for the test set using a trained model.

    The config dictionary should contain:

    ``test_csv``: Path to the test CSV.
    ``tokenizer_path``: Path to the tokenizer.
    ``model_ckpt``: Path to the model checkpoint to load.
    ``batch_size``: Batch size for inference.
    ``device``: Optional device specification.

    Returns
    -------
    List[str]
        A list of answer letters (``"A"``, ``"B"``, ``"C"`` or ``"D"``) of
        length equal to the number of rows in the test CSV.  The ordering
        matches the order of rows in the input file.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = VQAModel(
        model_ckpt_path=cfg.get("model_ckpt"),
        num_classes=int(cfg.get("num_classes", 4)),
        tokenizer_path=cfg.get("tokenizer_path"),
        device=device,
    )
    model.to(device)
    model.eval()
    dataset = Visual7WDataset(
        csv_path=cfg["test_csv"],
        tokenizer_path=cfg.get("tokenizer_path"),
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 16)), shuffle=False)
    predictions: List[str] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = [x.to(device) for x in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)
            for p in preds.cpu().tolist():
                predictions.append(chr(ord("A") + int(p)))
    return predictions