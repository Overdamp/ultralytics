# 🧠 EX26: The Adam Optimizer

Adaptive Moment Estimation (Adam) is one of the most widely used optimization algorithms in deep learning. It computes adaptive learning rates for each individual parameter by tracking exponential moving averages of both past gradients and past squared gradients.

---

## 1. Mathematical Formulation

Adam combines the principles of **Momentum** (first moment) and **RMSProp** (second moment).

For each parameter weight $w$:

### Step 1: Update Exponentially Decaying Average of Gradients (First Moment)
This acts like velocity, keeping the updates moving in the same direction:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

Where:
*   $g_t$ is the gradient at time step $t$.
*   $\beta_1$ is the decay rate for the first moment (default: `0.9`).

### Step 2: Update Exponentially Decaying Average of Squared Gradients (Second Moment)
This scales learning rates based on the magnitude of recent updates:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Where $\beta_2$ is the decay rate for the second moment (default: `0.999`).

### Step 3: Bias Correction
Because $m_t$ and $v_t$ are initialized as vectors of zeros, they are biased toward zero, especially during early training steps. We correct this bias:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Where $\beta_1^t$ and $\beta_2^t$ represent $\beta_1$ and $\beta_2$ raised to the power of the current time step $t$.

### Step 4: Parameter Update
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Where:
*   $\alpha$ is the learning rate.
*   $\epsilon$ (Epsilon) is a tiny constant (default: $10^{-8}$) preventing division by zero.

---

## 2. Comparing Adam vs. SGD
*   **SGD (Stochastic Gradient Descent):** Uses a single learning rate for all parameters. It is slower to converge but can lead to slightly better final model generalization when tuned perfectly with momentum.
*   **Adam:** Adapts the learning rate for *each individual weight*. Parameters with sparse, rare updates get larger steps, while parameters with frequent updates get smaller steps. It is robust, easy to configure, and converges quickly.

---

## 🔬 Computer Vision Connection: AdamW in YOLO
YOLO training utilizes **AdamW** (Adam with Decoupled Weight Decay). Standard Adam blends L2 regularization directly into the moving average calculations ($v_t$), causing weight decay to decay weights inproportionately. AdamW decouples weight decay, applying L2 shrinkage directly to the parameter value after the Adam update step.

---
*Related Topics:*
*   [[EX25_RMSProp\|EX25: RMSProp]]
*   [[EX27_Learning_Rate\|EX27: Learning Rate Scheduling]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
