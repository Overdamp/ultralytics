# 🧠 EX32: Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network (ANN). It consists of an input layer, one or more hidden layers of neurons, and an output layer.

---

## 1. MLP Architecture
An MLP is fully connected (dense), meaning every neuron in layer $l$ is connected to every neuron in the preceding layer $l-1$:

```
[ Input Vector x ] ---> [ Hidden Layer (a_1) ] ---> [ Hidden Layer (a_2) ] ---> [ Output Layer y_hat ]
```

---

## 2. Vectorized Forward Propagation Equations

To compute predictions in a neural network, we express the math in vectorized matrix form. For a network with one hidden layer:

### Layer 1 (Hidden Layer):
1.  **Linear Combination:** Multiply input vector $\mathbf{x}$ by weight matrix $\mathbf{W}^{[1]}$ and add bias vector $\mathbf{b}^{[1]}$:
    $$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$$
2.  **Activation Output:** Pass the sum through a non-linear activation function $g^{[1]}$:
    $$\mathbf{a}^{[1]} = g^{[1]}(\mathbf{z}^{[1]})$$

### Layer 2 (Output Layer):
1.  **Linear Combination:** Use the output of the hidden layer $\mathbf{a}^{[1]}$ as the input to the next layer:
    $$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$
2.  **Final Prediction ($\mathbf{\hat{y}}$):** Pass through output activation $g^{[2]}$ (e.g., Softmax for classification, Linear for regression):
    $$\mathbf{\hat{y}} = \mathbf{a}^{[2]} = g^{[2]}(\mathbf{z}^{[2]})$$

---

## 💡 The Universal Approximation Theorem
This theorem states that a feedforward network with a **single hidden layer** containing a finite number of neurons can approximate any continuous function, provided the activation function is non-linear. 
*   **The Power of Non-Linearity:** If we remove the activation functions, multiple linear layers collapse mathematically into a single linear layer, restricting the network to modeling simple straight lines. Non-linear activations allow MLPs to learn complex curved boundaries.

---

## 💻 Python PyTorch MLP Implementation

```python
import torch
import torch.nn as nn

# Define an MLP Model:
# - Input size = 4 features
# - Hidden layer = 8 neurons (with ReLU activation)
# - Output layer = 2 classes (logits)
mlp = nn.Sequential(
    nn.Linear(in_features=4, out_features=8),
    nn.ReLU(),
    nn.Linear(in_features=8, out_features=2)
)

# Mock input data (Batch size = 1, Features = 4)
x = torch.tensor([[1.5, -0.5, 2.0, 0.0]])

# Forward pass
output = mlp(x)
print(f"Input Tensor:  {x}")
print(f"Output Logits: {output}")
```

---
*Related Topics:*
*   [[EX31_Perceptron\|EX31: The Perceptron]]
*   [[EX33_Activation_Function\|EX33: Activation Functions]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
