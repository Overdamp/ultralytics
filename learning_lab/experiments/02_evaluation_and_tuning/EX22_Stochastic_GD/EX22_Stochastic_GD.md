# 🧠 EX22: Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent (SGD) is an optimization algorithm that updates the model's weights after evaluating a **single, randomly selected** training example. This introduces stochastic (random) noise that speeds up computation and helps escape local minima.

---

## 1. Why SGD? (The Batch GD Problem)
In standard **Batch Gradient Descent**, the model must calculate the gradients for *every single training sample* in the dataset before making a single weight update.
*   If your dataset contains 1,000,000 images, one single step of gradient descent requires 1,000,000 forward and backward passes.
*   This is computationally prohibitive and slows down training.

SGD solves this by calculating the gradient and updating weights immediately after processing **one** sample.

---

## 2. Mathematical Formulation

For a single selected training example $(x^{(i)}, y^{(i)})$, we define the single-sample loss:

$$J^{(i)}(w, b) = \frac{1}{2} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$

The gradient update is performed immediately:

$$w \leftarrow w - \alpha \left( h_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)}$$
$$b \leftarrow b - \alpha \left( h_{w,b}(x^{(i)}) - y^{(i)} \right)$$

Where:
*   $\alpha$ is the learning rate.
*   $h_{w,b}(x^{(i)}) - y^{(i)}$ is the prediction error.

---

## 3. The Role of Stochastic Noise
Because each individual training sample yields a noisy approximation of the true gradient of the whole dataset, the weights take a jagged, erratic path (resembling a random walk) toward the minimum:

```
Loss Surface
  \
   \     /\     /\     /\
    \___/  \___/  \___/  \---> Minimum (Noisy path helps jump over flat plateaus)
```

*   **Regularization:** This noise acts as a regularizer, preventing the model from fitting too early to noise, and helping it "jump" out of shallow local minima or saddle points.
*   **The Convergence Problem:** Because of the noise, SGD will never settle exactly at the global minimum. Instead, it will bounce around it.
*   **The Solution:** We must gradually reduce the learning rate over time (using schedulers like Cosine Annealing) so that the steps become smaller and the model settles down near the minimum.

---

## 💻 Python SGD Loop Implementation (From Scratch)

```python
import numpy as np

# Mock inputs (X) and targets (y)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([[2.0], [4.0], [6.0], [8.0]]) # True relationship: y = 2 * x

# Initialize parameters
w = 0.0
b = 0.0
lr = 0.01

# Run SGD for 5 epochs
for epoch in range(5):
    # Randomly shuffle data indices to ensure stochastic property
    indices = np.random.permutation(len(X))
    for idx in indices:
        xi = X[idx]
        yi = y[idx]
        
        # Forward pass (single sample prediction)
        prediction = w * xi + b
        
        # Compute single sample gradient
        error = prediction - yi
        dw = error * xi
        db = error
        
        # Immediate update
        w -= lr * dw
        b -= lr * db
        
    print(f"Epoch {epoch+1}: Weight = {w[0]:.4f}, Bias = {b[0]:.4f}")
```

---
*Related Topics:*
*   [[EX21_Gradient_Descent\|EX21: Gradient Descent]]
*   [[EX23_Mini_Batch_GD\|EX23: Mini-Batch Gradient Descent]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
