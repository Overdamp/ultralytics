# 🧠 EX29: Batch Size Analysis

Batch size is a training hyperparameter that defines the number of training samples processed by the neural network in a single forward/backward pass before updating the model's internal weights.

---

## 1. Trade-Offs: Small vs. Large Batch Sizes

Adjusting the batch size alters the training speed, gradient accuracy, and regularization:

### A. Small Batch Sizes (e.g., 2, 4, 8, 16)
*   **Gradient Noise:** Gradients are computed on few images, making them highly noisy (stochastic).
*   **Regularization Effect:** This noise acts as a regularizer, preventing the model from fitting too tightly to the training data and helping it escape sharp local minima, often leading to better generalization.
*   **Memory Efficiency:** Consumes very little GPU VRAM.
*   **Downside:** Takes longer to complete an epoch because weight updates happen more frequently.

### B. Large Batch Sizes (e.g., 64, 128, 256)
*   **Gradient Stability:** Gradients are highly accurate representations of the entire dataset.
*   **Speed:** Maximizes GPU parallelization, leading to very fast training epochs.
*   **Downside:** Requires massive GPU memory. Can cause the model to settle in flat local minima, sometimes leading to slightly lower generalization accuracy.

---

## 2. Learning Rate Scaling Rules

When you change the batch size, you must adjust the learning rate ($\alpha$). The most common guideline is the **Linear Scaling Rule**:
*   If you increase the batch size by a factor of $k$, you should increase the initial learning rate (`lr0`) by a factor of $k$.

$$\text{Learning Rate}_{\text{new}} = \text{Learning Rate}_{\text{base}} \times \left( \frac{\text{Batch Size}_{\text{new}}}{\text{Batch Size}_{\text{base}}} \right)$$

This is because with larger batches, there are fewer weight updates per epoch, so each update needs to be larger to cover the same distance.

---

## 🔬 Computer Vision Connection: CUDA OOM Errors
In YOLO training, setting the batch size too high for your GPU VRAM results in the notorious **CUDA Out of Memory (OOM)** error:
```text
OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

### How to Fix OOM:
1.  **Reduce Batch Size:** Decrease `batch=32` to `batch=16` or `batch=8`.
2.  **Reduce Image Size:** Decrease input resolution `imgsz=640` to `imgsz=320` (as memory scales quadratically with image resolution: $O(W \times H)$).
3.  **Accumulate Gradients:** Set batch size small, but update weights only after multiple batches (gradient accumulation).

```python
from ultralytics import YOLO

model = YOLO('yolo26n.pt')

# Training with batch size of 8 to prevent CUDA OOM
model.train(
    data='datasets/coco8/overall-ptt-object-detection.v11i.yolov11/data.yaml',
    epochs=10,
    batch=8,       # Force batch size to 8
    imgsz=640
)
```

---
*Related Topics:*
*   [[EX28_Weight_Decay\|EX28: Weight Decay]]
*   [[EX30_Epoch_Analysis\|EX30: Epoch Analysis]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
