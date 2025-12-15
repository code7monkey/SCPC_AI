# Visual Question Answering with BEiT‑3

This repository demonstrates how to organise a visual question answering (VQA) project built on top of the BEiT‑3 model into a clean, modular structure suitable for use on GitHub.  The code in this repository was originally developed in a pair of Jupyter notebooks and has been refactored into Python modules and scripts.

The goal of this project is to fine‑tune a multimodal language–vision model to answer multiple‑choice questions about images.  The repository is divided into a `src` package containing the core implementation, configuration files in `configs` controlling training and inference, and top–level scripts to orchestrate the training and inference workflows.

## Directory structure

```
vqa_project/
├── src/                 # Core Python package
│   ├── __init__.py      # Makes `src` importable and exposes key classes
│   ├── dataset.py       # Dataset definitions and helpers
│   ├── losses.py        # Loss functions
│   ├── model.py         # Model wrapper
│   ├── trainer.py       # Training and inference routines
│   └── utils.py         # Common utility functions
├── configs/
│   ├── train.yaml       # Experiment configuration for training
│   └── submit.yaml      # Configuration for inference and submission
├── train.py             # Script to run training
├── inference.py         # Script to run inference and create submission
├── requirements.txt     # Python dependencies
├── assets/
│   ├── model.pt         # Placeholder for the fine‑tuned model weights
│   └── tokenizer/       # Placeholder directory for the sentencepiece model
├── data/
│   ├── train.csv        # Pre‑processed training data (placeholder)
│   └── test.csv         # Test data (placeholder)
├── .gitignore           # Git ignore patterns
└── .gitattributes       # Git attributes, e.g. for large file support
```

## Getting started

Install the dependencies listed in `requirements.txt` into a virtual environment.  You will need PyTorch, the Hugging Face transformers library, and the usual data science stack:

```bash
pip install -r requirements.txt
```

Prepare your dataset by converting the raw Visual7W JSON file into a CSV format.  The script in `src/utils.py` defines a `make_mcq_json` function that converts the tabular data into the JSON format consumed by the BEiT‑3 training script.  Once your CSV and JSON files are ready, adjust the paths in `configs/train.yaml` accordingly.

### Training

To fine‑tune the BEiT‑3 model on your dataset, run

```bash
python train.py --config configs/train.yaml
```

The training script reads the YAML configuration, fixes random seeds for reproducibility, splits the training CSV into training and validation splits, writes the corresponding JSON files, and finally launches the training routine defined in `src/trainer.py`.  The default implementation wraps a simple PyTorch training loop, but you can replace it with a call to the original BEiT‑3 finetuning script if desired.

### Inference

After training, you can generate predictions on the test set by running

```bash
python inference.py --config configs/submit.yaml
```

This script loads the fine‑tuned model, runs inference on the test CSV, performs a simple majority vote if multiple checkpoints are provided, and writes the results to a CSV file in the format expected for submission.

## Notes

* The `assets/model.pt` and `assets/tokenizer/` placeholders must be replaced by actual model weights and sentencepiece files before training or inference will work.  These files are usually provided by the BEiT‑3 release and can be downloaded separately.
* The repository uses plain PyTorch for the training loop as an illustrative example.  If you prefer to reuse the original `run_beit3_finetuning.py` script provided by Microsoft, you can invoke it from within `src/trainer.py` using `subprocess.run` and pass the appropriate command‑line arguments defined in your YAML configuration.
* The `data/train.csv` and `data/test.csv` files included here are placeholders.  Please generate these files from your raw dataset before running the scripts.

We hope this structure helps you organise your work and simplifies collaboration when working with larger code bases.