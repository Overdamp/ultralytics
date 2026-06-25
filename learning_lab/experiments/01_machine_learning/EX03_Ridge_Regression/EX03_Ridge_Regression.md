# 🧠 EX03: Ridge Regression Theory (L2 Regularization)

In this topic, we examine **Ridge Regression**, a regularization technique used to prevent overfitting by adding an L2 penalty to the loss function.

---

## 1. Core Concept
When a regression model has too many features, or when those features are highly correlated (**Multicollinearity**), the weights ($w$) can become extremely large. This causes the model to fit noise in the training data, leading to **Overfitting** (high variance).

Ridge Regression solves this by penalizing large weights. It shrinks the weights towards zero, smoothing out predictions and helping the model generalize to unseen data.

---

## 2. Mathematical Formulation

### A. The Cost Function with L2 Penalty
Ridge regression modifies the Mean Squared Error (MSE) cost function by adding the sum of the squared weights multiplied by a tuning parameter $\lambda$ (Lambda):

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} w_j^2$$

Where:
*   $\lambda \ge 0$ is the **regularization parameter** (penalty strength).
    *   If $\lambda = 0$, Ridge Regression behaves exactly like Ordinary Least Squares (OLS) Linear Regression.
    *   As $\lambda \to \infty$, the weights $w \to 0$ (but never become exactly zero).
*   Note that the bias term $b$ is **not** penalized.

### B. Gradient Descent & Weight Decay
When we take the partial derivative of the cost function with respect to $w_j$, the update rule becomes:

$$w_j \leftarrow w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x_j^{(i)} + 2 \lambda w_j \right]$$

We can rearrange the terms to see the shrinking effect:

$$w_j \leftarrow w_j (1 - 2 \alpha \lambda) - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x_j^{(i)}$$

Since $(1 - 2 \alpha \lambda)$ is slightly less than 1, the weights are shrunk (decayed) by a fraction before the standard gradient step is applied. This mathematically proves why L2 regularization is also known as **Weight Decay**!

---

## 🛠️ Connection to Computer Vision & YOLO
*   **Optimizer Weight Decay:** Modern deep neural network optimizers (like AdamW or SGD with weight decay used to train YOLO models) utilize the exact same L2 regularization math. Penalizing large weights prevents neurons from relying too heavily on specific pixels, resulting in much higher robustness to shifts in lighting, noise, and backgrounds.

---
*Related Topics:*
*   [[EX02_Polynomial_Regression\|EX02: Polynomial Regression]]
*   [[EX04_Lasso_Regression\|EX04: Lasso Regression]]
*   Return to main syllabus: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
