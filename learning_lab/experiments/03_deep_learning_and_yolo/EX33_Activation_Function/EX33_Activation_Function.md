# 🧠 EX33: Activation Functions

Activation functions are mathematical operations applied to the outputs of neural network nodes. They introduce non-linearity into the network, enabling it to learn complex, non-linear patterns.

---

## 1. Why Activation Functions are Required
Without non-linear activation functions, multiple linear layers (e.g., in an MLP or CNN) collapse mathematically into a single linear operation:

$$\mathbf{W}^{[2]} \left( \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} \right) + \mathbf{b}^{[2]} = \mathbf{W}_{\text{effective}} \mathbf{x} + \mathbf{b}_{\text{effective}}$$

No matter how many hundreds of layers a network has, without activations, it can only model simple straight lines (solving only linear problems).

---

## 2. Common Activation Functions

### A. Sigmoid
Maps inputs to a probability range between 0 and 1:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

*   **Disadvantage:** Suffers from **vanishing gradients** at extreme inputs, slowing or stopping backpropagation.

### B. Rectified Linear Unit (ReLU)
Replaces negative values with zero, keeping positive values identical:

$$f(x) = \max(0, x)$$

*   **Advantage:** Extremely fast to compute, prevents vanishing gradients.
*   **Disadvantage:** "Dying ReLU" problem, where inactive neurons ($x < 0$) output 0 gradient forever.

### C. Leaky ReLU
Allows a small, non-zero gradient when the input is negative to prevent dying neurons:

$$f(x) = \max(0.01x, x)$$

### D. Sigmoid Linear Unit / Swish (SiLU - YOLO Default)
A smooth, self-gated activation function:

$$f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

*   **Why YOLO Uses SiLU:** Its curve is smooth and continuously differentiable (unlike the sharp corner of ReLU at $x=0$). It also has a small negative valley near $x < 0$, which yields smoother gradients during backpropagation, significantly boosting training convergence for deep detection networks.

```
Act. Output
  ^
  |        / (SiLU)
  |       /
  |  ____/
  | /    
--v-----------------> Input (x)
```

---

## 💻 Python PyTorch Activation Demonstration

```python
import torch
import torch.nn as nn

# Input tensor with negative and positive values
x = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])

# Initialize Activations
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.1)
silu = nn.SiLU()

print(f"Input:      {x.numpy()}")
print(f"ReLU:       {relu(x).numpy()}")
print(f"Leaky ReLU: {leaky_relu(x).numpy()}")
print(f"SiLU:       {silu(x).detach().numpy()}")
```

---
*Related Topics:*
*   [[EX32_MLP\|EX32: Multi-Layer Perceptron (MLP)]]
*   [[EX34_Backpropagation\|EX34: Backpropagation]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
