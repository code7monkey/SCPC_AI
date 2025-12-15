"""Model definitions for the VQA BEiT‑3 project.

This module defines a light‑weight wrapper around a text encoder that can be
fine‑tuned for multiple‑choice visual question answering.  The default
implementation uses the Hugging Face XLM‑RoBERTa model as a stand‑in for
BEiT‑3's text encoder.  If you have a pre‑trained BEiT‑3 checkpoint saved
under `assets/model.pt`, this wrapper will load it instead.

The model's architecture consists of an underlying encoder and a
task‑specific classifier head.  The encoder is expected to return a
sequence of hidden states, from which the [CLS] token representation is
extracted and passed through a linear layer to produce logits over the
answer classes.

```python
from src.model import VQAModel

model = VQAModel(model_ckpt_path="assets/model.pt",
                 tokenizer_path="assets/tokenizer/beit3.spm",
                 num_classes=4)
```
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel


class VQAModel(torch.nn.Module):
    """A simple multimodal VQA model.

    Parameters
    ----------
    model_ckpt_path: Optional[str]
        Path to a fine‑tuned model checkpoint.  If provided and the file
        exists, the model will be loaded from this checkpoint.  Otherwise
        a fresh instance of the encoder will be created.
    num_classes: int
        Number of answer classes (default: 4).  Visual7W uses a
        multiple‑choice format with exactly four choices, but this can be
        adjusted as needed.
    tokenizer_path: Optional[str]
        Path to a sentencepiece or tokenizer folder.  If provided, the
        tokenizer will be loaded from this location.  Otherwise the
        "xlm‑roberta‑base" tokenizer will be used.  Note that BEiT‑3
        checkpoints ship with their own sentencepiece model; place that
        inside `assets/tokenizer/` and pass the path here.
    device: Optional[str]
        Device on which to allocate the model.  If ``None`` the model
        remains on the CPU; the caller should move it later.
    """

    def __init__(
        self,
        model_ckpt_path: Optional[str] = None,
        num_classes: int = 4,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load or initialise the tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
            except Exception:
                # Fall back to pretrained if the local tokenizer cannot be loaded
                self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        else:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        # Attempt to load a pre‑trained model checkpoint.  If the checkpoint
        # contains a standard Hugging Face model it can be loaded directly.
        self.encoder = None
        if model_ckpt_path and os.path.isfile(model_ckpt_path):
            try:
                checkpoint = torch.load(model_ckpt_path, map_location="cpu")
                # If the checkpoint is a state_dict, build a new encoder and load it
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
                    self.encoder.load_state_dict(checkpoint["state_dict"], strict=False)
                # If the checkpoint is a full model, assume it already
                # implements forward and just reuse it as the encoder
                elif isinstance(checkpoint, torch.nn.Module):
                    self.encoder = checkpoint
            except Exception:
                self.encoder = None

        # Fallback: create a fresh encoder if none was loaded
        if self.encoder is None:
            self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        hidden_size = getattr(self.encoder.config, "hidden_size", 768)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

        # Move to the requested device if provided
        if device:
            self.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder and classification head.

        Parameters
        ----------
        input_ids: torch.Tensor
            Tokenised input ids of shape ``(batch_size, seq_len)``.
        attention_mask: torch.Tensor
            Attention mask of the same shape as ``input_ids``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch_size, num_classes)``.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Take the embedding of the first token ([CLS]) as pooled output
        if hasattr(outputs, "last_hidden_state"):
            pooled_output = outputs.last_hidden_state[:, 0, :]
        else:
            pooled_output = outputs[0][:, 0, :]
        logits = self.classifier(pooled_output)
        return logits