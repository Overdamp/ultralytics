# 🧠 EX54: Resuming Interrupted YOLO Training

Deep learning training runs can take hours, days, or even weeks. Interruptions due to power outages, out-of-memory errors, system updates, or cloud instance preemption are common occurrences. Ultralytics YOLO features a robust checkpointing system that allows developers to resume training from the exact state it was interrupted, without losing progress.

---

## 1. What Happens During Training Interruptions?

During a training run, YOLO saves two primary checkpoint files in your run weight directory (typically `runs/detect/train/weights/`):
*   **`best.pt`**: The model weights that achieved the highest evaluation metric (e.g., highest mAP) on the validation set so far.
*   **`last.pt`**: The complete state of the training session at the end of the most recent completed epoch.

### Inside `last.pt`: The Training State Dictionary
A simple weights file only contains the model parameters ($\theta$). However, `last.pt` is a serialized PyTorch state dictionary structure containing:

$$\mathbf{S}_{checkpoint} = \{ \Theta_t, \mathbf{\Phi}_{opt}, \Psi_{sched}, t, \mathbf{C}_{cfg} \}$$

Where:
1.  **Model Weights ($\Theta_t$):** The network weights at the end of epoch $t$.
2.  **Optimizer State ($\mathbf{\Phi}_{opt} = \{ \mathbf{m}_t, \mathbf{v}_t \}$):** The running gradients and momentum states for the optimizer.
3.  **Scheduler State ($\Psi_{sched}$):** The exact position on the learning rate decay curve (e.g., Cosine Annealing), preventing the learning rate from resetting to `lr0`.
4.  **Epoch Counter ($t$):** The index of the last completed epoch.
5.  **Training Configurations ($\mathbf{C}_{cfg}$):** The original hyperparameters (dataset path, batch size, image size, augmentations, etc.) used to start the training run.

---

## 2. Theoretical Foundations: Why Simple Weight Loading is Insufficient

Simply loading weights via `YOLO('last.pt')` and calling `model.train()` without `resume=True` resets the optimizer and scheduler state. This causes severe training instability.

### A. Optimizer Momentum Discontinuity (Gradient Shock)
Consider SGD with momentum. The update step is defined as:

$$v_t = \beta v_{t-1} + \eta g_t$$
$$\theta_t = \theta_{t-1} - v_t$$

If training is interrupted at epoch $k$ and resumed by resetting the optimizer, the velocity history is lost ($v_k = 0$). The new update step becomes:

$$v_{k+1} = \eta g_{k+1}$$

Since $v_k = 0$, the momentum term $\beta v_k$ is eliminated. This causes a sudden, massive direction change in parameter updates (gradient shock), leading to a spike in training loss and temporary divergence until the optimizer rebuilds its momentum state.

### B. Learning Rate Reset Shock
If the scheduler state is not recovered, the learning rate resets to $\eta_0$ (`lr0`). If the model weights are already partially optimized (e.g. at epoch 80 out of 100), applying a high initial learning rate will destroy the delicate features learned in later epochs:

$$\theta_{k+1} = \theta_k - \eta_0 g_{k+1} \quad \text{instead of} \quad \theta_{k+1} = \theta_k - \eta_k g_{k+1} \quad (\text{where } \eta_k \ll \eta_0)$$

By using `resume=True`, YOLO guarantees that the scheduler starts exactly at $\eta_k$, maintaining smooth learning decay.

---

## 3. Resume Parameter Details

When resuming, the primary arguments passed to the system are:

| Parameter | Type | Default Value | Range / Choices | Deep Technical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `model` | `str` | *Required* | Path to file | Must point specifically to the `last.pt` file in the runs weight directory. |
| `resume` | `bool` | `False` | `True`, `False` | Instructs the framework to load the model, optimizer, scheduler, and original hyperparameters from the checkpoint instead of starting a new run. |

> [!IMPORTANT]
> When calling `model.train(resume=True)`, do **not** pass other training parameters (like `epochs`, `data`, or `batch`). YOLO automatically reads all original parameters from the `last.pt` checkpoint to ensure continuity.

---

## 4. Python vs. CLI Code Comparison

### Python API Usage
```python
from ultralytics import YOLO

# 1. Load the checkpoint of the last saved epoch
model = YOLO('runs/detect/train/weights/last.pt')

# 2. Resume training
model.train(resume=True)
```

### CLI Usage
```bash
# Resume training using the resume command
yolo train resume model=runs/detect/train/weights/last.pt
```

---

## 💡 Professor Tips

### Overriding Hyperparameters on Resume (e.g. Out-of-Memory Recovery)
If your run crashed due to an Out-of-Memory (OOM) error and you need to resume with a smaller batch size, you cannot change `batch` in the arguments because `resume=True` ignores new argument inputs. 

To override settings on resume:
1.  Navigate to the training directory (e.g., `runs/detect/train/`).
2.  Open `args.yaml` in a text editor.
3.  Modify the parameters (e.g., change `batch: 16` to `batch: 8`).
4.  Execute the resume command. YOLO reads the updated parameters directly from this `args.yaml` configuration!

### Distributed Data Parallel (DDP) Resumption
When resuming training on a multi-GPU system (DDP), execute the resume command with the same number of GPUs originally used:
```bash
yolo train resume model=runs/detect/train/weights/last.pt device=[0,1]
```
Changing the number of GPUs mid-training requires recalculating gradient accumulation scales, which can alter training dynamics.

---
*Related Topics:*
*   [[EX53_Training_Settings|EX53: Training Settings]]
*   [[EX55_Evaluation_Modes|EX55: Evaluation Modes]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
