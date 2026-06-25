# 🧠 EX05: Logistic Regression Theory

In this topic, we examine **Logistic Regression**, which is the fundamental baseline algorithm for binary classification tasks.

---

## 1. Core Concept
Although it has "regression" in its name, Logistic Regression is used for **Classification**. It models the probability that an input $x$ belongs to a default class ($y=1$). 

Instead of fitting a straight line, it passes the linear output through a squashing function called the **Sigmoid (or Logistic) function**, restricting the output values strictly between $0$ and $1$.

---

## 2. Mathematical Formulation

### A. The Sigmoid Function
The sigmoid function $\sigma(z)$ maps any real-valued number to a probability value between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### B. The Hypothesis Function
We define the linear combination of inputs $z = \mathbf{w}^T \mathbf{x} + b$, and pass it through the sigmoid function:

$$\hat{y} = h_{\mathbf{w}}(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Where $\hat{y} = P(y=1 | x)$ represents the probability that the given example belongs to class 1.

### C. Cost Function: Binary Cross-Entropy (Log Loss)
We cannot use MSE for logistic regression because the sigmoid makes the cost function non-convex (multiple local minima). Instead, we use **Binary Cross-Entropy Loss** (Log Loss):

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

*   If $y^{(i)} = 1$, the second term becomes 0, penalizing predictions further from 1.
*   If $y^{(i)} = 0$, the first term becomes 0, penalizing predictions further from 0.

---

## 🔬 Computer Vision Case Study: Gauge Classification
Suppose YOLO detects a gauge component in your PTT dataset ([overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11)), and you want to classify if it is an `analog-gauge` ($y=1$) or a `digital-gauge` ($y=0$) based on:
1.  `x1`: Bounding box aspect ratio (analog gauges are usually square/circular: 1.0, digital are rectangular).
2.  `x2`: Text character density (using OCR).

Logistic regression will output a probability:
$$P(\text{analog-gauge}) = \sigma(w_1 \cdot \text{aspect\_ratio} + w_2 \cdot \text{character\_density} + b)$$

If the output probability is $\ge 0.5$, we classify it as an `analog-gauge`.

---
*Related Topics:*
*   [[EX01_Linear_Regression\|EX01: Linear Regression]]
*   [[EX06_KNN\|EX06: K-Nearest Neighbors]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
