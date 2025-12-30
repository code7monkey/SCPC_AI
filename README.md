# Samsung Collegiate Programming Challenges

**VQA Project – BEiT-3 기반 멀티모달 질문응답 모델**  
**Demo Baseline Code – Vision–Language Model Fine-tuning**

---

This repository provides a **baseline implementation for a Visual Question Answering (VQA) task** using a **BEiT-3 multimodal model**.  
It is intended for demonstration and educational purposes, focusing on **image–text joint modeling and fine-tuning workflows**.

The codebase is organized in a **modular and extensible structure**, with **YAML-based configuration files** to manage training and inference without modifying source code.

---

## 🎯 Project Goals

- **Multimodal input processing** – jointly handle images and textual questions
- **Fine-tuning BEiT-3** – adapt a pretrained vision–language model to a VQA dataset
- **Modular code design** – datasets, models, losses, and training loops separated under `src/`
- **YAML-based experiment management** – control training and inference via `configs/train.yaml` and `configs/submit.yaml`

---

## 📁 Project Structure

    vqa_project/
    ├── src/
    │   ├── __init__.py          # Package initialization and exports
    │   ├── dataset.py           # VQA dataset loading and preprocessing
    │   ├── losses.py            # Loss functions (e.g., cross entropy)
    │   ├── model.py             # BEiT-3 model wrapper (HuggingFace-based)
    │   ├── trainer.py           # Training and inference routines
    │   └── utils.py             # Common utilities (e.g., seed fixing)
    │
    ├── train.py                 # Training entry script
    ├── inference.py             # Inference & submission generation script
    │
    ├── configs/
    │   ├── train.yaml           # Training configuration
    │   └── submit.yaml          # Inference / submission configuration
    │
    ├── assets/
    │   ├── model.pt             # Fine-tuned model weights (placeholder)
    │   └── tokenizer/           # Tokenizer files (placeholder)
    │
    ├── data/
    │   ├── train.csv            # Training data (placeholder)
    │   └── test.csv             # Test data (placeholder)
    │
    ├── requirements.txt         # Required Python packages
    ├── .gitignore               # Git ignore patterns
    └── .gitattributes           # Git attributes (e.g., LFS settings)

---

## 🛠 Environment Setup

Python **3.9+** is recommended.

    pip install -r requirements.txt

The `requirements.txt` file includes core dependencies such as **PyTorch**, **HuggingFace Transformers**, and **pandas**.  
When using a GPU, make sure to install a **CUDA-compatible version of PyTorch**.

---

## 🚀 Usage

### Training

To fine-tune the BEiT-3 model, run:

    python train.py --config configs/train.yaml

The training script performs the following steps:

- Loads training configuration from the YAML file
- Fixes random seeds for reproducibility
- Loads and preprocesses CSV-based VQA data via `src/dataset.py`
- Calls the training loop implemented in `src/trainer.py` to fine-tune the model

By default, a simple PyTorch training loop is used.  
If needed, this can be extended to use **HuggingFace Trainer** or to call **official BEiT-3 fine-tuning scripts** via `subprocess`.

---

### Inference

To generate predictions on the test dataset:

    python inference.py --config configs/submit.yaml

The inference script:

- Loads the fine-tuned model from `assets/model.pt`
- Loads the tokenizer from `assets/tokenizer/`
- Preprocesses the test CSV data
- Runs inference using routines defined in `src/trainer.py`
- Outputs predictions in the required submission format

Ensembling multiple checkpoints can be enabled and configured directly in the YAML file.

---

## 📜 Notes

- `assets/model.pt` and `assets/tokenizer/` are provided as empty placeholders.  
  Download the pretrained BEiT-3 weights and tokenizer files and place them here before training.
- `data/train.csv` and `data/test.csv` are example placeholders.  
  For actual experiments, preprocess datasets such as **Visual7W** into CSV format and replace these files.
- All dataset paths, hyperparameters, and output directories are defined in `configs/train.yaml` and `configs/submit.yaml`.
- Large files such as datasets and checkpoints are excluded via `.gitignore`.  
  If needed, `.gitattributes` can be updated to manage large files using **Git LFS**.
