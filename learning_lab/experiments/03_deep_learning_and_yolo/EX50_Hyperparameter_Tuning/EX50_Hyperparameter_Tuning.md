# 🧠 EX50: Hyperparameter Tuning for YOLO

Hyperparameter tuning is the process of finding the optimal configuration of training parameters (e.g., learning rates, augmentation coefficients, regularization settings) to maximize a model's validation performance (mAP).

---

## 1. Core Concept
Unlike model parameters (weights and biases) which are learned automatically during training, **hyperparameters** must be set before training begins. Because YOLO models contain dozens of interdependent hyperparameters (spanning loss weights, optimization steps, and image augmentations), manual tuning can be extremely difficult.

---

## 2. Key YOLO Hyperparameters to Tune

### A. Optimization & Learning Rates
*   **`lr0` (Initial Learning Rate):** Determines the initial step size for weight updates (default: `0.01` for SGD, `0.001` for Adam). If too high, gradients explode; if too low, training gets stuck in local minima.
*   **`lrf` (Final Learning Rate Fraction):** The final learning rate is calculated as `lr0 * lrf`. This determines how much the learning rate shrinks by the end of training.
*   **`momentum` (Gradient Momentum):** Acceleration factor for weight updates (default: `0.937`).
*   **`weight_decay` (L2 Regularization):** Penalty to prevent weights from growing excessively large (default: `0.0005`).

### B. Warmup Hyperparameters
At the start of training, gradients can be highly unstable. YOLO uses a **warmup phase** (typically the first 3 epochs) where it gradually increases learning rates and scales momentum:
*   `warmup_epochs` (default: `3.0`)
*   `warmup_momentum` (default: `0.8`)
*   `warmup_bias_lr` (default: `0.1` - initial warmup learning rate for bias parameters)

### C. Data Augmentation Coefficients
Data augmentations prevent overfitting by artificially increasing dataset diversity:
*   **`mosaic` (default: `1.0`):** Combines 4 training images into one, forcing the model to learn small object features and reduce reliance on global context.
*   **`mixup` (default: `0.0`):** Overlays two images with different transparencies.
*   **`hsv_h`, `hsv_s`, `hsv_v`:** Random adjustments to Hue, Saturation, and Value to simulate varying lighting conditions.

---

## 3. Mathematical Learning Rate Scheduling (Cosine Annealing)

YOLO decays the learning rate $\eta_t$ at epoch $t$ using a **Cosine Annealing** schedule:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{cur}}{T_{\max}}\pi\right)\right)$$

Where:
*   $\eta_{\max}$ is the initial learning rate `lr0`.
*   $\eta_{\min}$ is the final learning rate target (`lr0 * lrf`).
*   $T_{cur}$ is the current training epoch.
*   $T_{\max}$ is the total number of epochs.

---

## 🔬 Computer Vision Case Study: Tuning for PTT Industrial Objects
When training YOLO on the [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11) dataset, we notice many classes (like `small-valve`, `lever-handle`, or `pig-alert`) are small and thin. 

### Custom Tuning Recommendations:
1.  **Reduce Spatial Augmentation:** High values of `scale` (e.g., scaling images down to 50%) might make tiny valves completely disappear. We should tune `scale` down to `0.3` or `0.2`.
2.  **Adjust Mosaic Closing:** Disabling mosaic augmentation earlier (`close_mosaic=15` or `20` epochs before the end) allows the model to fine-tune on clean, un-distorted object boundaries, significantly boosting final localization accuracy.

---

## 🛠️ Automated Tuning (Genetic Algorithms)
Ultralytics YOLO provides a built-in search mode that uses a **Genetic Algorithm (GA)**. It runs multiple short training cycles, mutates the best-performing hyperparameters, and converges on the optimal parameters automatically:

```python
from ultralytics import YOLO

model = YOLO('yolo26n.pt')
# Tune hyperparameters for 30 iterations, training for 10 epochs each
model.tune(data='datasets/coco8/overall-ptt-object-detection.v11i.yolov11/data.yaml', 
           epochs=10, 
           iterations=30, 
           optimizer='AdamW')
```

---
*Related Topics:*
*   [[EX49_YOLO_Result_Analysis\|EX49: YOLO Result Analysis]]
*   [[EX27_Learning_Rate\|EX27: Learning Rate]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
