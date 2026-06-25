# 🧠 EX11: Accuracy in Machine Learning

Accuracy is the most intuitive and commonly used evaluation metric for classification models. It measures the ratio of correct predictions to the total number of input samples.

---

## 1. Core Definitions (The Confusion Matrix Elements)
To understand accuracy, we must define the four outcomes of binary classification:
*   **True Positive (TP):** The model correctly predicts the positive class (e.g., correctly predicting a bounding box contains a `control-valve`).
*   **True Negative (TN):** The model correctly predicts the negative class (e.g., correctly predicting a background image has no object).
*   **False Positive (FP):** The model incorrectly predicts the positive class (e.g., predicting background clutter is a `control-valve`). Also known as a **Type I Error**.
*   **False Negative (FN):** The model incorrectly predicts the negative class (e.g., missing an actual `control-valve` and predicting background). Also known as a **Type II Error**.

---

## 2. Mathematical Formulation

The formula for accuracy is:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

It represents the probability that a randomly chosen prediction from the model is correct.

---

## ⚠️ The Accuracy Paradox (Class Imbalance)
Accuracy is a highly misleading metric when the dataset is **imbalanced** (one class far outnumbers the other). 

### Example Scenario:
Suppose you are training a model to detect gas pipe leaks using images.
*   **Total Images:** 10,000 images.
*   **Actual Leaks (Positive Class):** 100 images ($1\%$).
*   **Normal Pipes (Negative Class):** 9,900 images ($99\%$).

If a dumb model predicts "Normal Pipe" for **every single image** without looking at the pixels:
*   $TP = 0$ (it finds no leaks)
*   $TN = 9,900$ (it correctly identifies all normal pipes)
*   $FP = 0$
*   $FN = 100$ (it misses all leaks)

$$\text{Accuracy} = \frac{0 + 9,900}{0 + 9,900 + 0 + 100} = \frac{9,900}{10,000} = 99\%$$

The model achieves **$99\%$ accuracy**, but it is completely useless because it fails to detect a single leak! Therefore, for imbalanced datasets, we must rely on metrics like [[EX13_Precision\|Precision]] and [[EX14_Recall\|Recall]].

---

## 💻 Python Implementation

Here is how you calculate accuracy using Python:

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Ground truth labels (1 = valve present, 0 = background)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

# Model predictions
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

# Method 1: Manual Numpy calculation
correct_predictions = np.sum(y_true == y_pred)
total_predictions = len(y_true)
manual_accuracy = correct_predictions / total_predictions
print(f"Manual Accuracy: {manual_accuracy:.2f}")  # Outputs 0.80

# Method 2: Scikit-Learn
sklearn_accuracy = accuracy_score(y_true, y_pred)
print(f"Sklearn Accuracy: {sklearn_accuracy:.2f}")  # Outputs 0.80
```

---
*Related Topics:*
*   [[EX12_Confusion_Matrix\|EX12: Confusion Matrix]]
*   [[EX13_Precision\|EX13: Precision]]
*   [[EX14_Recall\|EX14: Recall]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
