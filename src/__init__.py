"""Top‑level package for the VQA BEiT‑3 project.

This module exposes the primary classes and functions for ease of import.  You
can simply import from `src` rather than reaching into individual files.  For
example:

```python
from src import VQAModel, Visual7WDataset, run_training, run_inference
```

See the individual submodules for full documentation.
"""

from .model import VQAModel
from .dataset import Visual7WDataset, load_dataframes, split_dataset
from .trainer import run_training, run_inference
from .utils import seed_everything, make_mcq_json

__all__ = [
    "VQAModel",
    "Visual7WDataset",
    "load_dataframes",
    "split_dataset",
    "run_training",
    "run_inference",
    "seed_everything",
    "make_mcq_json",
]