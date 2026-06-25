# 🧠 EX27: Learning Rate Scheduling

The learning rate (often denoted as $\alpha$ or $\eta$) is the most critical hyperparameter in training neural networks. It determines the step size taken towards the minimum of the loss function during optimization.

---

## 1. The Impact of Learning Rate Size

*   **Learning Rate Too High:** The model takes steps that are too large, causing the loss to oscillate or even **diverge** (the model weights explode, and loss becomes `NaN`).
*   **Learning Rate Too Low:** The model takes tiny steps. Training becomes extremely slow, and the model is highly likely to get stuck in poor local minima or saddle points.

To balance exploration and exploitation, we use **Learning Rate Schedulers** to dynamically adjust the step size over time.

---

## 2. Common Scheduling Strategies

### A. Step Decay
Reduces the learning rate by a multiplicative factor (e.g., $0.1$) after a fixed number of epochs (e.g., every 30 epochs).

### B. Cosine Annealing (YOLO Default)
Reduces the learning rate from its maximum initial value (`lr0`) to a minimum final value (`lr0 * lrf`) following a cosine curve:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{cur}}{T_{\max}}\pi\right)\right)$$

This allows the model to explore early, and converge smoothly into the global minimum during later epochs.

---

## 3. Learning Rate Warmup
At the beginning of training, weights are randomly initialized. A large learning rate can cause massive, erratic weight updates that ruin pre-trained features.

To prevent this, YOLO uses **Linear Warmup**:
*   For the first few epochs (typically `warmup_epochs=3.0`), the learning rate starts near zero (defined by `warmup_bias_lr` for bias terms) and increases linearly up to the target `lr0`.

```
Learning Rate
  |             / \
  |            /   \
  |           /     \_________________
  |          /                        \
  |_________/__________________________\____
  0       Warmup    Epochs             Total Epochs
```

---

## 💻 Python PyTorch Learning Rate Scheduling Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define a Cosine Annealing Scheduler
# - T_max: Number of epochs to complete the cosine cycle (e.g. 50 epochs)
# - eta_min: Minimum learning rate (final epoch target)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

# Simulated training loop
for epoch in range(50):
    # Train step (dummy loss)
    loss = model(torch.randn(1, 10)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Update learning rate
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Learning Rate = {current_lr:.6f}")
```

---
*Related Topics:*
*   [[EX26_Adam\|EX26: The Adam Optimizer]]
*   [[EX28_Weight_Decay\|EX28: Weight Decay]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
