# 🧠 EX53: YOLO Training Settings & Hyperparameters

Training a YOLO model is not just a matter of executing a command; it requires a deep understanding of how optimization settings, hardware utilization, learning rate scheduling, and dataset ingestion interact. Getting these settings right directly affects training speed, GPU memory footprint, and final model generalization.

---

## 1. Mathematical and Theoretical Foundations of YOLO Training

To design effective training configurations, we must analyze the underlying mathematical formulations of the optimization process.

### A. Optimizer Formulations: SGD vs. AdamW

YOLO supports multiple optimizers, with Stochastic Gradient Descent (SGD) with Momentum and Decoupled Weight Decay Adam (AdamW) being the primary choices.

#### Stochastic Gradient Descent (SGD) with Momentum
SGD updates parameters by computing the gradient over mini-batches. To reduce oscillation in steep ravines, momentum accumulates a velocity vector in the direction of persistent gradients:

$$v_t = \beta v_{t-1} + \eta g_t$$
$$\theta_t = \theta_{t-1} - v_t$$

Where:
*   $\theta$ represents the network weights.
*   $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ is the gradient of the loss function $\mathcal{L}$.
*   $\eta$ is the learning rate (`lr0`).
*   $\beta$ is the momentum coefficient (`momentum`, typically set to $0.937$ in YOLO).

#### AdamW (Decoupled Weight Decay)
For complex datasets, AdamW is often preferred. Unlike standard Adam, which applies L2 regularization directly to the gradients, AdamW decouples weight decay from the gradient updates to maintain correct scale-free learning rates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

Where:
*   $m_t$ and $v_t$ are the first and second biased moment estimates.
*   $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected estimates.
*   $\beta_1$ and $\beta_2$ are decay rates (typically $0.9$ and $0.999$).
*   $\lambda$ is the decoupled weight decay factor (`weight_decay`, typically $0.0005$).

### B. Learning Rate Scheduling: Cosine Annealing and Warmup
Instead of a step-decay schedule, YOLO utilizes Cosine Annealing to smoothly decay the learning rate to a minimum value `lrf` (fraction of `lr0`):

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$

Where:
*   $T_{cur}$ is the current epoch index.
*   $T_{max}$ is the total number of epochs.
*   $\eta_{max} = \text{lr0}$, and $\eta_{min} = \text{lr0} \times \text{lrf}$.

#### Warmup Phase
During the early phases of training (defined by `warmup_epochs`), the learning rate climbs gradually from a lower value (`warmup_bias_lr` or `warmup_lr`) to `lr0`. This stabilizes early gradient updates when weights are randomly initialized or adapted:
$$\eta_t = \eta_{start} + \left(\frac{t}{T_{warmup}}\right) (\text{lr0} - \eta_{start})$$

### C. Multi-Scale Training Mechanics
To improve the model's scale invariance, YOLO supports dynamic resolution scaling during training (when `rect=False`). Every 10 batches, the network selects a new image size randomly from a range around the target `imgsz`:

$$\text{imgsz}_{batch} = 32 \times \text{round}\left( \frac{\text{imgsz} \times r}{32} \right) \quad \text{where } r \sim \mathcal{U}(0.5, 1.5)$$

This forces the network to learn both low-level fine structures (from larger images) and global spatial configurations (from smaller images).

### D. Gradient Accumulation & Effective Batch Size
If physical GPU memory constraints prevent using a large batch size $B$, YOLO uses Gradient Accumulation. Gradients are computed over micro-batches and accumulated for $N_{accumulate}$ steps before executing `optimizer.step()`, creating an effective batch size:

$$B_{eff} = B \times N_{accumulate}$$

---

## 2. Comprehensive Parameter Details

The `model.train()` function supports a large array of configuration options:

| Parameter | Type | Default Value | Range / Choices | Deep Technical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `data` | `str` | *Required* | Path to file | Path to `data.yaml` config defining splits, classes `nc`, and class names. |
| `epochs` | `int` | `100` | $[1, \infty)$ | Total training passes over the dataset. |
| `batch` | `int` | `16` | $[-1, \infty)$ | Physical batch size. `-1` triggers auto-batching to maximize VRAM utilization. |
| `imgsz` | `int` | `640` | Multiples of 32 | Target resolution input. Must be a multiple of 32 due to downsampling operations. |
| `device` | `str/int/list` | `None` | `cpu`, `0`, `[0, 1]` | Execution target. Numeric values specify GPU IDs. Lists trigger Distributed Data Parallel (DDP). |
| `workers` | `int` | `8` | $[0, \infty)$ | Number of CPU worker threads for background data loader prep (augmentations, resizing). |
| `optimizer` | `str` | `'auto'` | `'SGD'`, `'Adam'`, `'AdamW'` | Choice of optimization algorithm. |
| `cos_lr` | `bool` | `False` | `True`, `False` | Activates cosine learning rate decay instead of linear decay. |
| `amp` | `bool` | `True` | `True`, `False` | Enables Automatic Mixed Precision (FP16 math) to halve VRAM usage and boost speed. |
| `lr0` | `float` | `0.01` | $(0, 1]$ | Initial learning rate for optimizer (SGD default: `0.01`, AdamW default: `0.001`). |
| `lrf` | `float` | `0.01` | $(0, 1]$ | Final learning rate scale factor ($\text{lr0} \times \text{lrf}$ is the target minimum rate). |
| `momentum` | `float` | `0.937` | $[0, 1)$ | Momentum coefficient for SGD or beta1 parameter for Adam family. |
| `weight_decay` | `float` | `0.0005` | $[0, 1)$ | Regularization factor preventing weight growth. |
| `warmup_epochs` | `float` | `3.0` | $[0, \infty)$ | Number of epochs spent gradually climbing from warmup learning rate to `lr0`. |
| `close_mosaic` | `int` | `10` | $[0, \infty)$ | Number of epochs at the end of training where the standard Mosaic augmentation is disabled. |

---

## 3. Python vs. CLI Code Comparison

### Python API Usage
```python
from ultralytics import YOLO

# Load a pre-trained backbone
model = YOLO('yolo26n.pt')

# Train using customized hyperparameters
results = model.train(
    data='datasets/custom_data/data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    device=0,
    workers=4,
    project='industrial_inspection',
    name='yolo26_valve_detection'
)
```

### CLI Usage
```bash
yolo train \
  model=yolo26n.pt \
  data=datasets/custom_data/data.yaml \
  epochs=50 \
  batch=16 \
  imgsz=640 \
  device=0 \
  workers=4 \
  project=industrial_inspection \
  name=yolo26_valve_detection
```

---

## 💡 Professor Tips: Avoiding GPU Out-of-Memory (OOM) Errors

When you hit a `RuntimeError: CUDA out of memory`, follow these remediation steps:

1.  **Reduce Batch Size:** VRAM scales linearly with batch size. Cut `batch` in half (e.g. $32 \rightarrow 16 \rightarrow 8$).
2.  **Reduce Input Resolution:** If batch reduction is insufficient, reduce `imgsz` (e.g., $640 \rightarrow 512 \rightarrow 384$). Always ensure it is a multiple of $32$.
3.  **Validate Mixed Precision:** Confirm `amp=True`. If disabled, you lose FP16 memory efficiency (VRAM usage doubles).
4.  **CPU RAM Exhaustion:** If your system crashes without a CUDA traceback, check your CPU RAM. High `workers` count creates multiple dataloader child processes, multiplying the memory usage of datasets loaded in RAM. Drop `workers` to $2$ or $0$.

---
*Related Topics:*
*   [[EX52_Model_Configurations|EX52: Model Configurations]]
*   [[EX54_Resuming_Training|EX54: Resuming Training]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
