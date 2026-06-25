# 🧠 EX13: Precision (Positive Predictive Value)

Precision measures the accuracy of a model's positive predictions. It answers: *Of all instances the model predicted as positive, how many were actually correct?*

---

## 1. Mathematical Formulation

Precision is defined as the ratio of True Positives ($TP$) to all predicted positives (which is the sum of True Positives and False Positives, $TP + FP$):

$$\text{Precision} = \frac{TP}{TP + FP}$$

Where:
*   **True Positive (TP):** The model correctly predicts the positive class (e.g., drawing a bounding box around a `flange` that is actually a flange).
*   **False Positive (FP):** The model incorrectly predicts the positive class (e.g., drawing a bounding box around a shadow or background clutter and labeling it a `flange`). Also known as a **false alarm** or **Type I Error**.
*   If the model makes zero false predictions ($FP = 0$), Precision is $1.0$ ($100\%$).

---

## 2. Alarm Fatigue & The Cost of False Positives
In real-world applications, a low-precision model causes **Alarm Fatigue**. If a security camera or industrial monitoring system sounds an alarm 100 times a day, but 99 of those times are false alarms (shadows, birds), human operators will begin to ignore the alarm. Eventually, when a real event occurs, it will be missed because the operators turned off the system.

---

## 🔬 Computer Vision Case Study: Automated Inspections
In your PTT dataset [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11), suppose you use YOLO to detect and count `wellheads` in drone images:

*   **When to Prioritize Precision:** 
    *   If sending a maintenance crew to inspect a flagged wellhead requires a helicopter ride or an 8-hour drive, **False Positives are extremely expensive**.
    *   **Decision:** You prioritize **Precision** by setting a high confidence threshold (e.g. `conf=0.75`). You only alert the crew when the model is extremely certain of a detection, accepting that you might miss a few obscured wellheads (lower Recall) in order to avoid expensive false trips.

---

## 💻 Python Precision Calculation

```python
import numpy as np
from sklearn.metrics import precision_score

# Ground truth binary labels (1 = flange present, 0 = background)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

# Model predictions
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])  # Misses 1 flange (FN), adds 1 false alarm (FP)

# 1. Manual calculation
tp = np.sum((y_true == 1) & (y_pred == 1))
fp = np.sum((y_true == 0) & (y_pred == 1))
manual_precision = tp / (tp + fp)
print(f"Manual Precision: {manual_precision:.4f}")  # Outputs 4 / 5 = 0.8000

# 2. Scikit-Learn
sklearn_precision = precision_score(y_true, y_pred)
print(f"Sklearn Precision: {sklearn_precision:.4f}")
```

---
*Related Topics:*
*   [[EX12_Confusion_Matrix\|EX12: Confusion Matrix]]
*   [[EX14_Recall\|EX14: Recall (Sensitivity)]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
