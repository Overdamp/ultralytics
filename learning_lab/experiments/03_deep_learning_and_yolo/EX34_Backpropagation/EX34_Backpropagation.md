# 🧠 EX34: Backpropagation & The Chain Rule

Backpropagation (backward propagation of errors) is the core algorithm used to train neural networks. It calculates the gradient of the loss function with respect to each weight and bias, allowing optimizers to update parameters and minimize error.

---

## 1. The Two-Pass Learning Cycle
1.  **Forward Pass:** The input features propagate forward through the network layers. Linear combinations and activation functions are applied, producing predictions and a final Loss value ($L$).
2.  **Backward Pass:** Starting at the output layer, the algorithm calculates the gradient of the Loss with respect to every weight and bias, passing these derivatives backward using the **Chain Rule**.

---

## 2. Mathematical Derivation (Single-Neuron Example)

Let's examine a simple single-input neuron:

```
[ Input: x ] ---> ( Linear Combine: z = w*x + b ) ---> ( Activation: a = f(z) ) ---> [ Prediction: a ] ---> ( Loss: L = (a - y)^2 )
```

Our goal is to compute $\frac{\partial L}{\partial w}$ (how a change in weight $w$ affects the loss $L$) to update $w$.

Applying the **Chain Rule**:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Step-by-Step Derivation:
1.  **Loss Gradient ($\frac{\partial L}{\partial a}$):** How the loss changes with the prediction $a$:
    $$\frac{\partial L}{\partial a} = \frac{\partial}{\partial a}(a - y)^2 = 2(a - y)$$
2.  **Activation Gradient ($\frac{\partial a}{\partial z}$):** How the prediction changes with the pre-activation sum $z$:
    $$\frac{\partial a}{\partial z} = \frac{\partial}{\partial z}f(z) = f'(z)$$
3.  **Linear Gradient ($\frac{\partial z}{\partial w}$):** How the sum changes with the weight $w$:
    $$\frac{\partial z}{\partial w} = \frac{\partial}{\partial w}(w \cdot x + b) = x$$

### Final Gradient:
Combining these three components gives:

$$\frac{\partial L}{\partial w} = 2(a - y) \cdot f'(z) \cdot x$$

---

## ⚠️ Challenges in Deep Networks: Vanishing & Exploding Gradients
In very deep networks (like YOLO, which can have over 200 layers), the chain rule multiplies derivatives across all layers:
*   **Vanishing Gradients:** If activation function derivatives are small (e.g., Sigmoid outputs max $0.25$), multiplying them repeatedly causes the gradient to shrink to 0 in early layers, stopping them from learning.
*   **Exploding Gradients:** If derivatives are large ($> 1$), repeated multiplications cause the gradient to grow exponentially, making the weights unstable.
*   **Solutions in YOLO:** YOLO uses **SiLU (Swish)** activations to prevent vanishing gradients and **Skip/Residual Connections** (C3k2/C2PSA modules) which allow gradients to flow directly backward without shrinking.

---

## 💻 Python PyTorch Autograd Demonstration

PyTorch automates backpropagation using its `autograd` engine:

```python
import torch

# Initialize weight, input, and target with gradient tracking enabled
x = torch.tensor([1.5])
w = torch.tensor([0.8], requires_grad=True)
b = torch.tensor([0.2], requires_grad=True)
y = torch.tensor([2.0])  # Target

# 1. Forward Pass
z = w * x + b
a = torch.sigmoid(z)  # Sigmoid activation
loss = (a - y) ** 2   # Squared error loss

# 2. Backward Pass
loss.backward()  # Calculates gradients automatically

# Output computed gradients: dL/dw and dL/db
print(f"Computed Gradient dL/dw: {w.grad.item():.4f}")
print(f"Computed Gradient dL/db: {b.grad.item():.4f}")
```

---
*Related Topics:*
*   [[EX33_Activation_Function\|EX33: Activation Functions]]
*   [[EX35_CNN\|EX35: Convolutional Neural Networks]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
