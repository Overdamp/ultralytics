# 🧠 EX01: Linear Regression Theory

Welcome to the first step of your machine learning journey! In this topic, we will examine **Linear Regression**, the most fundamental algorithm used to predict continuous numerical values (regression task).

---

## 1. Core Concept
Linear regression assumes a linear relationship between the input features (independent variables $x$) and the output target (dependent variable $y$). 

Our goal is to find the best-fitting straight line (known as the **regression line**) that minimizes the error between our model's predictions and the actual ground-truth values.

---

## 2. Mathematical Formulation

### A. The Hypothesis Function
For a single input feature $x$, the predicted value $\hat{y}$ (also written as $h(x)$) is formulated as:

$$\hat{y} = h_{w,b}(x) = w \cdot x + b$$

Where:
*   $w$ represents the **Weight** (or coefficient/slope). It controls the angle of the regression line.
*   $b$ represents the **Bias** (or intercept). It represents the value of $y$ when $x = 0$.

In vector notation for multiple input features:
$$\hat{y} = h_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

### B. Cost Function: Mean Squared Error (MSE)
To measure how "wrong" our model predictions are, we define a cost function. For linear regression, we use the **Mean Squared Error (MSE)** cost function:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$

Where:
*   $m$ is the total number of training examples.
*   $x^{(i)}$ and $y^{(i)}$ are the input and true target values for the $i$-th training example.
*   The factor of $\frac{1}{2}$ is added for mathematical convenience during differentiation (canceling out the exponent of $2$ when calculating gradients).

### C. Optimization: Gradient Descent
To minimize the cost $J(w, b)$, we adjust $w$ and $b$ iteratively using **Gradient Descent** by moving in the opposite direction of the gradients:

$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

Calculating the partial derivatives yields:

$$w \leftarrow w - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)}$$
$$b \leftarrow b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right)$$

Where $\alpha$ (Alpha) is the **Learning Rate**, controlling the step size we take towards the minimum.

---

## 🛠️ Connection to Computer Vision & YOLO
*   While YOLO uses deep neural networks, the concept of predicting continuous bounding box coordinates ($x, y, w, h$) is essentially a regression task! 
*   Understanding how weights change to match targets forms the bedrock of backpropagation in deep neural networks.

---
*Related Topics:*
*   [[EX02_Polynomial_Regression\|EX02: Polynomial Regression]]
*   [[EX21_Gradient_Descent\|EX21: Gradient Descent]]
*   Return to main syllabus: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
