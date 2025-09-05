# VGG 16 and ResNet 18 comparison for image classification

This repository contains a **from‑scratch PyTorch implementation** and comparison of two classic CNN architectures for image classification:

- **VGG‑16** (deep, plain convolutional blocks)
- **ResNet‑18** (with residual/skip connections)

The notebook trains both models on an **ImageFolder** dataset with standard data augmentation and reports accuracy and diagnostic metrics.

> Primary artifact: `main code.ipynb`

## What’s inside
- Custom implementations: `VGG16`, `ResidualBlock`, and `ResNet18` classes.
- Training utilities with **CrossEntropyLoss**, **Adam, SGD** optimizer(s), and a **StepLR** scheduler.
- Data pipeline using `torchvision.transforms` (Resize, RandomRotation, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize).
- Evaluation helpers (overall accuracy, confusion matrix, classification report).

## Quick start

### 1) Create a Python environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** For GPU training, install the CUDA‑enabled builds of **torch**/**torchvision** following the official PyTorch instructions for your OS/driver.

### 3) Prepare the dataset
This notebook expects an **ImageFolder** layout. Put your images into class‑named subfolders:

```
data/
  train/
    class_a/ img1.jpg ...
    class_b/ ...
    ...
  val/
    class_a/ ...
    class_b/ ...
  test/
    class_a/ ...
    class_b/ ...
```

Update any `data_root` or dataloader paths in the notebook if your structure differs.

### 4) Run the notebook
```bash
jupyter notebook "main code.ipynb"
```

## Training configuration (auto‑detected)
- Batch size: 64
- Epochs: 40
- Optimizer(s): Adam, SGD
- Learning rate(s): 0.001, 0.01
- LR scheduler: `StepLR(step_size=5, gamma=0.5)`

## Results
Example results printed by the notebook include:
Test Accuracy: 94.49%, Test Accuracy: 96.38%

You can also render a confusion matrix and classification report (via scikit‑learn) to inspect per‑class performance.

## Repository structure
```
.
├── main code .ipynb
├── README.md
└── requirements.txt
```
## Reproducibility tips
- Fix a random seed and set `torch.backends.cudnn.deterministic = True` for stable runs.
- Save your trained weights and metrics after each epoch to `outputs/`.
- Log learning curves (loss/accuracy) to compare architectures fairly.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
