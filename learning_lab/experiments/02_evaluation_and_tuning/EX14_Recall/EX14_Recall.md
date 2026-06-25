# 🧠 EX14: Recall (Sensitivity)

Recall (also known as Sensitivity or True Positive Rate) measures the completeness of a classifier's positive predictions. It answers: *Of all actual positive instances in the dataset, how many did the model successfully find?*

---

## 1. Mathematical Formulation

Recall is defined as the ratio of True Positives ($TP$) to all actual positive instances (which is the sum of True Positives and False Negatives, $TP + FN$):

$$\text{Recall} = \frac{TP}{TP + FN}$$

Where:
*   **True Positive (TP):** The model correctly detects an object (e.g. correctly draws a bounding box on a `control-valve`).
*   **False Negative (FN):** The model misses an object (e.g. fails to detect an actual `control-valve`, leaving it unlabeled). 
*   If the model misses zero objects ($FN = 0$), Recall is $1.0$ ($100\%$).

---

## 2. The Precision-Recall Trade-Off
We can easily achieve $100\%$ Recall by predicting that **every** region in an image contains an object. However, this will result in thousands of False Positives (clutter labeled as objects), causing **Precision to drop to near $0\%$**.

To increase Recall, we lower the confidence threshold (`conf`), making the model more sensitive. To increase Precision, we raise the threshold, making the model more selective.

---

## 🔬 Computer Vision Case Study: Safety Critical Systems
When designing a vision pipeline for industrial monitoring (such as your PTT dataset [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11)):

*   **When to Prioritize Recall:** If detecting safety hazards (like a pressure release valve stuck shut, or a structural crack):
    *   **False Positive (FP):** Model alerts a hazard, but it was just a shadow. Cost: A technician performs a quick manual check.
    *   **False Negative (FN):** Model misses a hazard. Cost: Equipment failure or pipe rupture.
    *   **Decision:** In this scenario, we prioritize **Recall** by setting a lower confidence threshold (e.g. `conf=0.25`) to guarantee we miss nothing, accepting a few false alarms.

---

## 💻 Python Recall Calculation

```python
import numpy as np
from sklearn.metrics import recall_score

# Ground truth binary labels (1 = valve present, 0 = background)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

# Model predictions
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])  # Misses 1 valve (FN), adds 1 false alarm (FP)

# 1. Manual calculation
tp = np.sum((y_true == 1) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))
manual_recall = tp / (tp + fn)
print(f"Manual Recall: {manual_recall:.4f}")  # Outputs 4 / 5 = 0.80

# 2. Scikit-Learn
sklearn_recall = recall_score(y_true, y_pred)
print(f"Sklearn Recall: {sklearn_recall:.4f}")
```

---
*Related Topics:*
*   [[EX13_Precision\|EX13: Precision]]
*   [[EX15_F1_Score\|EX15: F1 Score]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
