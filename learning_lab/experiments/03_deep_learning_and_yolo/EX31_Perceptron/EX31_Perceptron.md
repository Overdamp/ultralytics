# 🧠 EX31: The Perceptron

Invented by Frank Rosenblatt in 1958, the Perceptron is the simplest form of an artificial neural network, modeled after the structure of a biological neuron. It serves as a linear binary classifier.

---

## 1. Mathematical Formulation

A Perceptron takes $n$ numeric inputs, multiplies them by corresponding weights, sums them with a bias, and passes the result through a step activation function.

```
[ Inputs: x1, x2, ..., xn ] ---> [ Weighted Sum: z = w*x + b ] ---> [ Heaviside Step Function f(z) ] ---> [ Output: y_hat (0 or 1) ]
```

### A. The Linear Combination (Weighted Sum)
$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

Where:
*   $\mathbf{x}$ is the input feature vector.
*   $\mathbf{w}$ is the weight vector.
*   $b$ is the bias term (shifting the activation threshold).

### B. The Heaviside Step Activation Function
The output prediction $\hat{y}$ is determined by a hard threshold:

$$\hat{y} = f(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$

---

## 2. The Perceptron Learning Rule
To train a Perceptron, weights and biases are updated iteratively whenever the model makes a prediction error.

For each training example $(\mathbf{x}, y)$:
1.  Compute prediction $\hat{y}$.
2.  Update weights and bias:

$$w_i \leftarrow w_i + \eta (y - \hat{y}) x_i$$
$$b \leftarrow b + \eta (y - \hat{y})$$

Where:
*   $y$ is the actual ground truth label ($0$ or $1$).
*   $\hat{y}$ is the predicted label ($0$ or $1$).
*   $\eta$ (Eta) is the learning rate ($0 < \eta \le 1$).
*   *Note:* If the prediction is correct ($y - \hat{y} = 0$), no update occurs.

---

## ⚠️ The Linear Separability Limit (The XOR Problem)
A single Perceptron can only classify datasets that are **linearly separable** (meaning the classes can be divided by a single straight line).
*   **Logical Gates:** A perceptron can easily solve `AND`, `OR`, and `NOT` gates.
*   **The XOR Gate:** A perceptron **cannot** solve `XOR` (Exclusive OR) because its classes cannot be separated by a straight line. 
*   This limitation (published by Minsky and Papert in 1969) proved that single-layer perceptrons were limited, triggering the first **AI Winter** until Multi-Layer Perceptrons (MLPs) with non-linear activation functions were developed.

---

## 💻 Python Perceptron Implementation (From Scratch)

Here is a scratch implementation of a Perceptron trained to solve an `OR` logic gate:

```python
import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=0.1):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
        self.lr = lr
        
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if z >= 0 else 0

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error

# Training data for OR Gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

p = Perceptron(input_dim=2)
p.train(X, y, epochs=10)

print("Trained Weights:", p.weights)
print("Trained Bias:", p.bias)
print("Prediction for [1, 0]:", p.predict([1, 0]))  # Outputs 1
```

---
*Related Topics:*
*   [[EX30_Epoch_Analysis\|EX30: Epoch Analysis]]
*   [[EX32_MLP\|EX32: Multi-Layer Perceptron (MLP)]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
