# 🧠 EX15: The F1 Score

The F1 Score is the harmonic mean of Precision and Recall. It provides a single balanced metric that is especially useful when evaluating classification models on imbalanced datasets.

---

## 1. Why the Harmonic Mean?
If we used the simple **Arithmetic Mean** to combine Precision ($P$) and Recall ($R$):

$$\text{Arithmetic Mean} = \frac{P + R}{2}$$

Consider a model that predicts positive for only $1$ sample, which happens to be correct, but misses $99$ other actual positive samples:
*   $\text{Precision} = 1.0$ (no false positives)
*   $\text{Recall} = 0.01$ (missed almost everything)
*   $\text{Arithmetic Mean} = \frac{1.0 + 0.01}{2} = 0.505$ ($50.5\%$ looks decent, which is highly misleading).

The **Harmonic Mean** takes the reciprocal of values, heavily penalizing extreme differences:

$$F_1 = \frac{2}{\frac{1}{P} + \frac{1}{R}} = 2 \cdot \frac{P \cdot R}{P + R}$$

Using the same values:
$$F_1 = 2 \cdot \frac{1.0 \cdot 0.01}{1.0 + 0.01} = \frac{0.02}{1.01} \approx 0.0198 \quad (1.98\%)$$

The F1 Score collapses to near $0\%$, accurately reflecting the model's poor recall performance.

---

## 2. Generalization: The F-Beta Score
In some scenarios, you care more about Precision (preventing false alarms) or Recall (preventing misses). We can adjust their relative weights using the **$F_\beta$ Score**:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

*   **$\beta = 1.0$:** Standard F1 Score (equal weight).
*   **$\beta = 0.5$:** Weighs Precision higher than Recall. Used when false positives are highly costly.
*   **$\beta = 2.0$:** Weighs Recall higher than Precision. Used when missing an object is highly dangerous (e.g. medical diagnosis or gas leaks).

---

## 🔬 Computer Vision Connection: YOLO F1 Curve
In your training results (e.g., [runs/detect/train-3/BoxF1_curve.png](file:///home/luke/ai_training/ultralytics/runs/detect/train-3/BoxF1_curve.png)):
*   YOLO plots the F1 Score (y-axis) against all possible confidence thresholds (x-axis).
*   The peak of this curve tells you the **optimal confidence threshold** to use during model deployment to get the best balance between Precision and Recall.

---

## 💻 Python F1 & F-Beta Score Calculation

```python
import numpy as np
from sklearn.metrics import f1_score, fbeta_score

# Ground truth binary labels
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

# Model predictions
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

# 1. Compute standard F1 Score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")  # Outputs 0.80

# 2. Compute F-Beta (Beta = 2.0, prioritizing Recall/reducing misses)
f2 = fbeta_score(y_true, y_pred, beta=2.0)
print(f"F-Beta (Beta=2): {f2:.4f}")
```

---
*Related Topics:*
*   [[EX14_Recall\|EX14: Recall]]
*   [[EX16_ROC_Curve\|EX16: ROC Curve]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
