# 🧠 EX17: Area Under the Curve (AUC)

AUC (Area Under the ROC Curve) measures the entire two-dimensional area underneath the ROC curve. It provides an aggregate measure of a classifier's performance across all possible classification thresholds.

---

## 1. Mathematical Formulation

AUC is mathematically defined as the integral of the True Positive Rate (TPR/Recall) with respect to the False Positive Rate (FPR):

$$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d\text{FPR}$$

Since $\text{TPR}$ and $\text{FPR}$ are bounded between $0$ and $1$, the AUC value is always bounded between $0$ and $1$:

| AUC Value | Interpretation |
| :--- | :--- |
| **$\text{AUC} = 1.0$** | **Perfect Classifier:** The model separates positive and negative classes with $100\%$ accuracy (no overlap between distributions). |
| **$0.5 < \text{AUC} < 1.0$** | **Realistic Classifier:** The model has a good degree of separability (higher is better). |
| **$\text{AUC} = 0.5$** | **Random Guessing:** The model has no discriminative ability (equivalent to flipping a coin). |
| **$\text{AUC} < 0.5$** | **Reciprocal Classifier:** The model is predicting the exact opposite classes (can be fixed by flipping predictions). |

---

## 2. Physical & Probabilistic Interpretation

Mathematically, AUC is equivalent to the probability that a classifier will rank a **randomly chosen positive instance** higher than a **randomly chosen negative instance**:

$$P\left( f(x^+) > f(x^-) \right)$$

For example, if a model's $\text{AUC} = 0.90$, it means that given a random positive image (e.g. an image containing a `lever-valve`) and a random negative image (e.g. background grass/pipes), there is a **$90\%$ chance** that the model will output a higher confidence score for the valve than the background.

---

## 3. Advantages of AUC
1.  **Scale-Invariant:** It measures how well predictions are **ranked** rather than their absolute values (e.g., if you multiply all predictions by 2, the AUC remains identical).
2.  **Classification-Threshold-Invariant:** It measures the model's quality overall, regardless of what specific threshold (e.g. `0.5` or `0.8`) is chosen to make final binary decisions.

---

## 💻 Python AUC Score Calculation

Here is how you compute the AUC score using Scikit-Learn:

```python
import numpy as np
from sklearn.metrics import roc_auc_score

# Ground truth binary labels (1 = positive class, 0 = negative class)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

# Raw prediction scores (probabilities output by the model)
y_scores = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.85, 0.4, 0.1, 0.65, 0.3])

# Calculate AUC Score
auc_score = roc_auc_score(y_true, y_scores)
print(f"AUC Score: {auc_score:.4f}")  # Outputs 0.96 (Excellent separability)
```

---
*Related Topics:*
*   [[EX16_ROC_Curve\|EX16: ROC Curve]]
*   [[EX18_Precision_Recall_Curve\|EX18: Precision-Recall Curve]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
