# 🧠 EX25: RMSProp Optimization

Root Mean Square Propagation (RMSProp) is an adaptive learning rate optimization algorithm proposed by Geoffrey Hinton. It resolves the vanishing or exploding learning rate problem by scaling weight updates by a running average of the recent squared gradients.

---

## 1. The AdaGrad Problem (Why RMSProp was invented)
To understand RMSProp, we must look at its predecessor, **AdaGrad**. AdaGrad scales the learning rate of each weight by dividing it by the sum of *all* historical squared gradients:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha}{\sqrt{\sum_{\tau=1}^{t} g_\tau^2} + \epsilon} g_t$$

*   **The Flaw:** Because squared gradients ($g_\tau^2$) are always positive, the sum in the denominator grows continuously throughout training. 
*   **The Result:** The learning rate eventually shrinks to exactly zero, shutting down learning before the model can reach the global minimum.

---

## 2. The RMSProp Solution
RMSProp solves this by replacing the cumulative sum with an **exponentially decaying average** of squared gradients. This means the optimizer "forgets" distant historical gradients and only scales updates based on recent gradient steps.

At each step $t$:

### Step 1: Compute Running Average of Squared Gradients ($v_t$)
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) g_t^2$$

Where:
*   $g_t^2$ is the element-wise square of the current gradient vector.
*   $\beta$ is the decay rate parameter (typically set to `0.9` or `0.99`), controlling how far back the memory window stretches.

### Step 2: Update Parameters
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha}{\sqrt{\mathbf{v}_t} + \epsilon} \odot g_t$$

Where:
*   $\alpha$ is the learning rate.
*   $\epsilon$ is a tiny smoothing term ($10^{-8}$) preventing division by zero.
*   $\odot$ is element-wise multiplication.

### The Effect:
*   If a weight has large gradients (oscillating rapidly), $v_t$ becomes large, which **divides** the learning rate and shrinks updates, dampening oscillations.
*   If a weight has small gradients (sluggish movement), $v_t$ becomes small, which **multiplies** the learning rate and increases updates, accelerating learning.

---

## 💻 Python PyTorch RMSProp Optimizer Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10, 2))

# Define RMSProp optimizer
# - lr: Learning rate (alpha)
# - alpha: Smoothing constant (equivalent to the decay factor beta, default 0.99)
# - eps: Epsilon (default 1e-8)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)
```

---
*Related Topics:*
*   [[EX24_Momentum\|EX24: Momentum Optimization]]
*   [[EX26_Adam\|EX26: The Adam Optimizer]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
