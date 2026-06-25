# 🧠 EX02: Polynomial Regression Theory

In this topic, we examine **Polynomial Regression**, an extension of linear regression used when the relationship between the independent variable $x$ and the dependent variable $y$ is non-linear.

---

## 1. Core Concept
In many real-world scenarios, data points do not follow a simple straight line. If we fit a straight line to curved data, our model will suffer from high bias (**Underfitting**). 

Polynomial Regression addresses this by adding power terms (e.g., $x^2, x^3$) as new features, allowing the regression line to bend and fit the curves of the dataset while still using the linear regression optimization framework.

---

## 2. Mathematical Formulation

### A. The Hypothesis Function
The relationship is modeled as an $n$-th degree polynomial:

$$\hat{y} = h_{w}(x) = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \dots + w_n x^n$$

Where:
*   $n$ is the **degree** of the polynomial.
*   $w_0$ is the bias/intercept.
*   $w_1, w_2, \dots, w_n$ are the weights for each respective power of $x$.

### B. Linear Transformation
Although the curve is non-linear with respect to $x$, the model is still **linear with respect to the weights ($w$)**. We can treat this as Multiple Linear Regression by mapping the power terms to new feature variables:

$$x_1 = x, \quad x_2 = x^2, \quad x_3 = x^3, \quad \dots, \quad x_n = x^n$$

Rewriting the hypothesis:
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + \dots + w_n x_n$$

This allows us to solve the polynomial regression problem using the exact same cost functions (MSE) and Gradient Descent updates as standard Linear Regression!

---

## 🛠️ Connection to Computer Vision & YOLO
*   **Object Tracking Trajectories:** When tracking moving objects in videos (e.g., a ball falling, a vehicle changing lanes, or a robot arm moving), their path over time is rarely a straight line. Polynomial regression is used in Kalman Filters or tracking algorithms to model and predict the curved future trajectories ($x(t), y(t)$) of bounding boxes.

---
*Related Topics:*
*   [[EX01_Linear_Regression\|EX01: Linear Regression]]
*   [[EX03_Ridge_Regression\|EX03: Ridge Regression]]
*   Return to main syllabus: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
