# 🧠 EX28: Weight Decay Optimization

Weight decay is a regularization technique that prevents overfitting by adding a penalty proportional to the size of the network weights, causing them to decay toward zero.

---

## 1. Mathematical Formulation

In standard Stochastic Gradient Descent (SGD), weight decay is mathematically equivalent to **L2 Regularization**.

### A. L2 Regularization Cost Function:
We add a penalty term to our base loss function $E_0(\mathbf{w})$:

$$E(\mathbf{w}) = E_0(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

Where $\lambda$ is the weight decay coefficient.

### B. Gradient Update Derivation:
Taking the gradient of the loss function with respect to weights:

$$\nabla E(\mathbf{w}) = \nabla E_0(\mathbf{w}) + \lambda \mathbf{w}$$

The parameter update step is:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla E(\mathbf{w}_t) = \mathbf{w}_t - \alpha \left( \nabla E_0(\mathbf{w}_t) + \lambda \mathbf{w}_t \right)$$

Rearranging terms:

$$\mathbf{w}_{t+1} = \mathbf{w}_t(1 - \alpha \lambda) - \alpha \nabla E_0(\mathbf{w}_t)$$

This proves that at each epoch, the weight is multiplied by a shrinkage factor $(1 - \alpha \lambda)$ (slightly less than 1) before subtracting the gradient of the raw loss.

---

## 2. Decoupled Weight Decay (AdamW)
While L2 regularization and Weight Decay are identical in SGD, they **diverge in adaptive optimizers** like Adam. 

In standard Adam, adding L2 regularization to the loss propagates the weight decay term through the moving average gradient estimators. This causes weights with large historical gradients to decay *less* than weights with small gradients, which is counterproductive.

**AdamW** resolves this by decoupling the weight decay step, applying it directly to the parameter value rather than adding it to the gradient calculation:

$$\mathbf{w}_{t+1} = \mathbf{w}_t(1 - \alpha \lambda) - \text{Adam\_Update}(\nabla E_0(\mathbf{w}_t))$$

Because YOLO defaults to using AdamW, this decoupled approach is essential for keeping training stable.

---

## 💻 Python PyTorch Optimizer Implementation

Here is how you specify weight decay in PyTorch optimizers:

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10, 2))

# Scenario A: SGD with Weight Decay (Equivalent to L2)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005)

# Scenario B: AdamW (Decoupled Weight Decay - YOLO Default)
# This keeps weight decay separate from the adaptive gradient moment estimators.
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
```

---
*Related Topics:*
*   [[EX27_Learning_Rate\|EX27: Learning Rate Scheduling]]
*   [[EX29_Batch_Size\|EX29: Batch Size Analysis]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
