# 🧠 EX24: Momentum Optimization

Momentum is an optimization technique that accelerates Gradient Descent by accumulating velocity in directions of consistent gradients, dampening oscillations, and helping the model escape local minima.

---

## 1. The Physical Analogy: A Rolling Ball
Standard Gradient Descent acts like a step-by-step walker who evaluates the slope at their feet and takes a step, forgetting their speed immediately. If the slope changes direction, they turn immediately.

**Momentum** acts like a heavy ball rolling down a hill:
*   As the ball rolls, it accumulates velocity from gravity (the gradient).
*   Its mass (momentum) prevents it from stopping immediately on flat sections, helping it roll past flat valleys and escape shallow local minima.

---

## 2. Dampening Oscillations in Ravines
In complex loss landscapes (like steep ravines), standard SGD oscillates back and forth across the steep walls rather than moving along the floor towards the minimum:

```text
SGD (Oscillating):                  Momentum (Smooth):
   \    /\    /\    /                  \              /
    \  /  \  /  \  /                    \------------> (Smoothly moves along the valley floor)
     \/    \/    \/                      \          /
```

With Momentum, the oscillating components of the gradient updates cancel each other out over time, while the components pointing along the valley floor reinforce each other, leading to faster, smoother convergence.

---

## 3. Mathematical Formulation

At each iteration $t$:

### Step 1: Calculate Velocity ($v_t$)
We add a fraction $\beta$ of the previous velocity to the current gradient update:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \alpha \nabla J(\mathbf{w}_t)$$

Where:
*   $\\beta \\in [0, 1]$ is the **momentum factor** (acting as friction). If $\beta = 0.9$, it means the current step retains $90\%$ of its previous speed.
*   $\alpha$ is the learning rate.
*   $\nabla J(\mathbf{w}_t)$ is the gradient.

### Step 2: Update Weights
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_t$$

---

## 💡 Nesterov Accelerated Gradient (NAG)
Nesterov Momentum is a "look-ahead" optimization. Instead of calculating the gradient at the current position $\mathbf{w}_t$, it calculates the gradient at the projected position $\mathbf{w}_t - \beta \mathbf{v}_{t-1}$:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \alpha \nabla J(\mathbf{w}_t - \beta \mathbf{v}_{t-1})$$

This acts as a smart braking mechanism. If the momentum is about to carry the model up an opposing slope, the look-ahead gradient detects the climb early and slows down the updates, preventing overshooting.

---

## 🔬 Computer Vision Connection: YOLO Optimizer Momentum
In YOLO training runs (such as in your `train-3` configuration logs), the SGD optimizer momentum defaults to `momentum=0.937`. This value is optimized to balance convergence speed and prevent the weights from diverging on custom datasets.

---
*Related Topics:*
*   [[EX23_Mini_Batch_GD\|EX23: Mini-Batch Gradient Descent]]
*   [[EX25_RMSProp\|EX25: RMSProp]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
