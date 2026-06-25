# 🧠 EX18: The Precision-Recall (PR) Curve

The Precision-Recall (PR) curve is a graphical plot illustrating the trade-off between a model's Precision (accuracy of predictions) and Recall (capture rate) across all possible confidence thresholds.

---

## 1. Why Use PR Curve Over ROC Curve in Computer Vision?

In standard classification, the ROC curve (True Positive Rate vs. False Positive Rate) is popular. However, in **Object Detection**, we almost always use the **PR Curve**. 

### The True Negative (TN) Problem:
*   An image consists of millions of background pixels/grid cells that do not contain any objects. 
*   If we calculate False Positive Rate:
    $$\text{FPR} = \frac{FP}{TN + FP}$$
    Because $TN$ (True Negatives) is extremely large, the denominator explodes, driving $\text{FPR}$ to near $0$. This makes the ROC curve look deceptively perfect, even if the detector is making many false predictions.
*   **PR Curve Solution:** The calculations for Precision and Recall do **not** use True Negatives ($TN$):
    $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
    This makes the PR curve highly sensitive to false alarms ($FP$) and missed objects ($FN$), representing detector performance accurately.

---

## 2. How to Interpret the PR Curve

*   **The Ideal Classifier:** A curve that reaches the top-right corner (representing $100\%$ Precision and $100\%$ Recall).
*   **The Trade-Off:**
    *   If you set a high confidence threshold: Precision is high, but Recall is low (few detections, but highly accurate).
    *   If you set a low confidence threshold: Recall is high, but Precision is low (detects everything, but includes many false alarms).
*   **Area Under Curve (AUC):** The area under the PR curve is mathematically defined as the **Average Precision (AP)** for that class:

$$\text{AP} = \int_{0}^{1} P(R) \, dR$$

---

## 🔬 Computer Vision Connection: YOLO PR Charts
In your training runs (e.g. [runs/detect/train-3/BoxPR_curve.png](file:///home/luke/ai_training/ultralytics/runs/detect/train-3/BoxPR_curve.png)):
*   The chart plots individual PR curves for each of your 26 classes (like `control-valve` or `flange`).
*   It also plots a bold cyan curve representing the **mAP@0.5** (mean AP) averaged across all classes.
*   The closer each class curve is to bowing out to the top-right, the better the model detects that specific object class.

---

## 💻 Python Precision-Recall Curve Plotter

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Ground truth binary labels (1 = object present, 0 = background)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

# Predicted probabilities (confidence scores) from YOLO
y_scores = np.array([0.9, 0.1, 0.8, 0.75, 0.2, 0.85, 0.4, 0.15, 0.7, 0.3, 0.95, 0.6, 0.05, 0.8, 0.25])

# Calculate Precision-Recall pairs
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

# Plotting the curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'PR Curve (AUC/AP = {pr_auc:.2f})', color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('learning_lab/experiments/02_evaluation_and_tuning/EX18_Precision_Recall_Curve/pr_curve_sample.png')
print("PR Curve plot saved successfully.")
```

---
*Related Topics:*
*   [[EX17_AUC\|EX17: AUC (Area Under Curve)]]
*   [[EX19_Cross_Validation\|EX19: Cross-Validation]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
