# 🧠 EX12: The Confusion Matrix

A Confusion Matrix is a performance evaluation table used in supervised machine learning. It displays correct predictions alongside model errors (false alarms, misses, and class confusions), making it easy to see if a model is mixing up specific classes.

---

## 1. Binary Confusion Matrix Layout
For binary classification, the matrix is a $2\times2$ grid:

| | Predicted Positive (1) | Predicted Negative (0) |
| :--- | :--- | :--- |
| **Actual Positive (1)** | **True Positive (TP):** Correctly predicted positive. | **False Negative (FN):** Missed detection (Type II Error). |
| **Actual Negative (0)** | **False Positive (FP):** False alarm (Type I Error). | **True Negative (TN):** Correctly predicted negative. |

---

## 2. Multi-Class Object Detection Extension (Background Folds)
In object detection, we classify multiple object categories (e.g., 26 classes in your PTT dataset) and must account for background pixels:
*   A **"background" class** is added as an extra row and column in the matrix.
*   **Predicted Background (row match):** If an actual `control-valve` is not detected by YOLO, it is counted in the `Background` column (a False Negative / missed object).
*   **Actual Background (column match):** If YOLO draws a bounding box around background metal clutter and labels it a `flange`, it is counted in the `Background` row (a False Positive / false alarm).

---

## 3. Normalized vs. Raw Confusion Matrix
*   **Raw Matrix:** Displays the absolute count of predictions (e.g. 45 times predicted as class A, 3 times as class B). This can be dominated by class counts.
*   **Normalized Matrix:** Divides each row by the sum of actual samples in that class (each row sums to `1.0` or `100%`). This displays the accuracy percentage for each class independently of dataset size.

---

## 🔬 Computer Vision Connection: YOLO Confusion Matrix
In your training run folder (e.g. [runs/detect/train-3/confusion_matrix_normalized.png](file:///home/luke/ai_training/ultralytics/runs/detect/train-3/confusion_matrix_normalized.png)):
*   The diagonal running from top-left to bottom-right shows correct predictions. You want these numbers to be close to `1.0`.
*   Look for off-diagonal values: if the cell at row `control-valve` and column `valve-body` is `0.20`, it means your model is misclassifying $20\%$ of control valves as valve bodies. This tells you that you need to gather more distinct training images for these two classes.

---

## 💻 Python Multi-Class Confusion Matrix Calculation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# True labels of 3 classes: 0=flange, 1=valve, 2=gauge
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 1])

# Model predictions
y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 1, 1, 2, 0, 2, 2, 1])  # Confuses gauge (2) as valve (1)

# Generate raw confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['flange', 'valve', 'gauge'])
disp.plot(cmap=plt.cm.Blues)
plt.title("PTT Multi-Class Confusion Matrix")
plt.savefig('learning_lab/experiments/02_evaluation_and_tuning/EX12_Confusion_Matrix/confusion_matrix_plot.png')
print("Confusion Matrix plot saved.")
```

---
*Related Topics:*
*   [[EX11_Accuracy\|EX11: Accuracy]]
*   [[EX13_Precision\|EX13: Precision (Positive Predictive Value)]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
