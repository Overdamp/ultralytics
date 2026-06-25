# 🧠 EX04: Lasso Regression Theory (L1 Regularization)

In this topic, we examine **Lasso Regression** (Least Absolute Shrinkage and Selection Operator), a regularization technique that adds an L1 penalty to the loss function, enabling built-in feature selection.

---

## 1. Core Concept
Unlike Ridge Regression ($L2$) which shrinks weights close to zero but keeps all features, Lasso ($L1$) regression shrinks less important feature weights **exactly to zero**. 

This makes Lasso incredibly powerful for:
1.  **Feature Selection:** Automatically identifying and keeping only the most predictive features.
2.  **Model Sparsity:** Creating simpler, more interpretable models with fewer non-zero coefficients.

---

## 2. Mathematical Formulation

### A. The Cost Function with L1 Penalty
Lasso modifies the Mean Squared Error (MSE) cost function by adding the sum of the absolute values of the weights multiplied by $\lambda$:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} |w_j|$$

Where:
*   $\lambda \ge 0$ is the regularization strength.
*   $|w_j|$ is the absolute value of the weight.
*   Like Ridge, the bias term $b$ is **not** penalized.

### B. Geometry of L1 vs. L2 Regularization
Why does L1 drive weights to exactly zero while L2 does not?
*   **L2 Constraint (Circle):** The L2 penalty constraint region is a sphere/circle ($w_1^2 + w_2^2 \le t$). The cost function contours usually touch this circle at a point where weights are small but non-zero.
*   **L1 Constraint (Diamond):** The L1 penalty constraint region is a diamond ($|w_1| + |w_2| \le t$) which has sharp corners on the coordinate axes. The cost contours are highly likely to hit these corners first, setting the weight of the corresponding axis feature to exactly zero.

---

## 🔬 Computer Vision Case Study: Bounding Box Classification
Suppose we want to classify objects in your PTT dataset (e.g., separating a `lever-valve` from a `control-valve` inside [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11)) based on bounding box characteristics:
1.  `x1`: Bounding box width
2.  `x2`: Bounding box height
3.  `x3`: Aspect ratio (width / height)
4.  `x4`: Pixel intensity mean (color contrast)
5.  `x5`: Image center coordinate (position in image)

### Applying Lasso:
If the location of the object in the frame (`x5`) and background contrast (`x4`) are irrelevant for distinguishing the valve types, Lasso regression will drive their coefficients to exactly zero:
*   $w_4 \to 0$
*   $w_5 \to 0$

Leaving a sparse model containing only shape descriptors:
$$\hat{y} = w_0 + w_1 \cdot \text{width} + w_2 \cdot \text{height} + w_3 \cdot \text{aspect\_ratio}$$

---
*Related Topics:*
*   [[EX03_Ridge_Regression\|EX03: Ridge Regression (L2)]]
*   [[EX05_Logistic_Regression\|EX05: Logistic Regression]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
