# AI Project Context: YOLO Object Detection

## Project Overview
This project is an environment built on top of the **Ultralytics YOLO** framework. The main objective is to train, evaluate, and export custom YOLO models (specifically utilizing `yolo26n.pt` architecture) for detecting custom "PTT objects".

## Key Components

### 1. Data (`datasets/`)
- **Location:** `datasets/coco8/overall-ptt-object-detection.v11i.yolov11/`
- **Configuration:** `data.yaml` defines the dataset paths and the specific PTT object classes. This is the primary dataset being used for fine-tuning.

### 2. Educational / Workflow Notebooks
The root directory contains Jupyter Notebooks that serve as a playbook for this project:
- `YOLO_Training_Manual.ipynb`: Guidelines for initiating training, managing datasets, and transfer learning.
- `YOLO_Metrics_Explained.ipynb`: Information on interpreting evaluation metrics like mAP, Precision, and Recall.
- `YOLO_Hyperparameters_Explained.ipynb`: Notes on configuring learning rates, batch sizes, and optimization techniques.

### 3. Models & Weights
- **Base Weights:** `yolo26n.pt` (PyTorch format) is used as the foundational lightweight model for transfer learning.
- **Exported Models:** `yolo26n.onnx` indicates that models are being exported to the ONNX format for production deployment, likely intended for the Triton Inference Server.

### 4. Training History (`runs/detect/`)
- Past training experiments and validation runs are saved here (e.g., `train/`, `train-2/`, `val/`). 
- These directories contain the resulting loss curves, validation metrics, and the best/last `.pt` weights for the respective experiments.

## Standard AI Assistant Workflow For This Project
1. **Training:** Help configure parameters or debug dataset issues via `data.yaml`.
2. **Analysis:** Review metrics from the `runs/` folder to check for overfitting/underfitting.
3. **Deployment:** Assist in exporting the trained models to `.onnx` or `.engine` formats.
4. **Documentation:** Update the Jupyter notebooks when a new technique or metric is explored.
