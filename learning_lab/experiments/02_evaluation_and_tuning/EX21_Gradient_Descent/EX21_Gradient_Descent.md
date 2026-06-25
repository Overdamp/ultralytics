# 🧠 EX21: Gradient Descent Optimization

Gradient Descent is a first-order iterative optimization algorithm used to minimize a differentiable cost function $J(\mathbf{w})$ by finding its local or global minimum.

---

## 1. Core Concept
Imagine standing on a foggy mountain peak and wanting to find the path to the valley floor. Because the fog blocks your vision, you can only examine the slope of the ground directly under your feet.
*   To descend as quickly as possible, you step in the direction of the **steepest downward slope**.
*   In optimization, the gradient vector $\nabla J(\mathbf{w})$ points in the direction of the steepest *ascent* (up the hill). 
*   Therefore, to find the minimum (the valley), we must move in the **opposite direction** (subtracting the gradient).

---

## 2. Mathematical Formulation

For a weight vector $\mathbf{w}$, the update rule is:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla J(\mathbf{w}_t)$$

Where:
*   $\mathbf{w}_t$ is the current weight vector.
*   $\mathbf{w}_{t+1}$ is the updated weight vector.
*   $\alpha > 0$ is the **Learning Rate** (step size).
*   $\nabla J(\mathbf{w}_t)$ is the **gradient vector** of partial derivatives:

$$\nabla J(\mathbf{w}) = \left[ \frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2}, \dots, \frac{\partial J}{\partial w_n} \right]^T$$

---

## 3. Convex vs. Non-Convex Loss Landscapes
*   **Convex Loss Function (e.g., Linear Regression OLS):** The loss surface looks like a bowl (has exactly one global minimum). Gradient descent is mathematically guaranteed to find the global minimum if the learning rate is configured correctly.
*   **Non-Convex Loss Function (e.g., Deep Neural Networks like YOLO):** The loss surface is highly complex, containing multiple local minima, flat plateaus, and **saddle points** (where the slope is zero but it is not a minimum). Basic gradient descent can easily get trapped in these areas, requiring adaptive optimizers (like Adam) or stochastic updates (like SGD) to escape.

---

## 💻 Python Gradient Descent Implementation

Here is a simple scratch implementation of Gradient Descent to find the minimum of the quadratic function $f(x) = x^2$ (analytical minimum is at $x=0$):

```python
# Function: f(x) = x^2
def cost_function(x):
    return x**2

# Derivative: df/dx = 2 * x
def compute_gradient(x):
    return 2 * x

# Parameters
x = 10.0  # Starting point
lr = 0.1  # Learning rate
epochs = 20

print("Starting Gradient Descent:")
for epoch in range(epochs):
    # Compute gradient (slope)
    gradient = compute_gradient(x)
    
    # Update position (move opposite to slope)
    x = x - lr * gradient
    
    cost = cost_function(x)
    print(f"Epoch {epoch+1}: x = {x:.4f}, Cost = {cost:.4f}")
```

---
*Related Topics:*
*   [[EX20_Bias_Variance\|EX20: Bias-Variance Tradeoff]]
*   [[EX22_Stochastic_GD\|EX22: Stochastic Gradient Descent]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
