# 🧠 EX49: YOLO Result Analysis

Analyzing YOLO training results is the process of examining output metrics, curves, and prediction images to diagnose model performance, detect overfitting, and plan parameter adjustments.

---

## 1. Structure of the YOLO Training Output Directory
When you train a YOLO model, a directory is created under `runs/detect/train*/` containing several files:

| File Name | Description | What to Look For |
| :--- | :--- | :--- |
| **`weights/best.pt`** | The model weight state that achieved the highest validation mAP. | Use this for deployment and inference testing. |
| **`weights/last.pt`** | The weight state from the final epoch. | Use this if you need to resume training from where it stopped. |
| **`results.csv`** | A row-by-row log of every training epoch. | Tracks epoch time, losses (box, class, dfl), learning rates, and validation metrics. |
| **`results.png`** | Plotted charts of all losses and metrics over epochs. | Look for smooth, downward loss curves and upward mAP curves. |
| **`confusion_matrix_normalized.png`** | Grid showing correct predictions on the diagonal and class confusions on the off-diagonal. | Look for diagonal values close to `1.0`. High off-diagonal values show class confusion. |
| **`BoxPR_curve.png`** | Precision-Recall curve plotted for all classes. | Shows the trade-off. An ideal curve bows out toward the top-right corner. |
| **`val_batch*_pred.jpg`** | Visual bounding box predictions on validation images. | Inspect visually to find box shifts, missed objects, or double detections. |

---

## 2. Diagnosing Training Behavior

### A. Overfitting
*   **Symptom:** Training losses (`train/box_loss`, `train/cls_loss`) continue to decrease steadily, but validation losses (`val/box_loss`, `val/cls_loss`) start climbing or flatten out.
*   **Solution:** Increase regularization (`weight_decay`), adjust augmentations (`mosaic`, `mixup`), or stop training early using smaller `epochs` or `patience` settings.

### B. Underfitting
*   **Symptom:** Both training and validation losses remain high or plateau early, and mAP values remain low.
*   **Solution:** Train for more epochs, use a larger model size (e.g., scale up from `yolo26n` to `yolo26s`), or lower regularization weights.

---

## 3. Plotting Training Progress with Pandas
You can parse the `results.csv` file using Python to visualize training and validation loss curves:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training logs
df = pd.read_csv('runs/detect/train-3/results.csv')

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()

# Plot Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Box Loss')
plt.legend()
plt.grid(True)
plt.savefig('learning_lab/experiments/03_deep_learning_and_yolo/EX49_YOLO_Result_Analysis/loss_curve_check.png')
print("Loss curve check chart saved successfully.")
```

---
*Related Topics:*
*   [[EX48_mAP\|EX48: Mean Average Precision (mAP)]]
*   [[EX50_Hyperparameter_Tuning\|EX50: Hyperparameter Tuning]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
