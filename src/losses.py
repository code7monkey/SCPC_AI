"""Loss functions for the VQA BEiT‑3 project.

This module defines modular loss functions which can be swapped out as
needed.  For multiple‑choice question answering a standard cross
entropy loss is appropriate.  If you add new tasks or metrics, place
their loss definitions here.
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """Wrapper around :class:`torch.nn.CrossEntropyLoss`.

    Exposes a call interface compatible with other losses.  By wrapping
    ``nn.CrossEntropyLoss`` we make it easy to extend or customise the
    behaviour later (e.g. label smoothing).
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, labels)