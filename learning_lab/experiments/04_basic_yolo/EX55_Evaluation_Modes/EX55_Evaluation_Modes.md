# 🧠 EX55: Evaluation & Validation Modes in YOLO

Evaluation is the process of measuring a trained model's performance on a validation dataset. While training focuses on minimizing loss, validation translates the model's predictions into human-interpretable computer vision metrics such as Precision, Recall, and Mean Average Precision (mAP).

---

## 1. Validation vs. Inference (Prediction)

It is vital to distinguish between **Validation** (`val`) and **Inference** (`predict`):
*   **Validation (`val`):** Evaluates the model on *labeled* data (the `val` split in `data.yaml`). Since ground-truth labels are available, YOLO can calculate overlap (IoU) and compile accuracy metrics.
*   **Inference (`predict`):** Runs the model on *unlabeled* data (new images/videos). Since there are no labels, YOLO only outputs coordinates and confidence scores without computing any metrics.

---

## 2. Mathematical and Theoretical Foundations of Object Detection Metrics

To truly evaluate a model, we must understand the mathematical formulations of the metrics computed during validation.

### A. Intersection over Union (IoU)
IoU measures the spatial overlap between a predicted bounding box $B_p$ and a ground-truth box $B_{gt}$:

$$\text{IoU} = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p) + \text{Area}(B_{gt}) - \text{Area}(B_p \cap B_{gt})}$$

### B. Defining TP, FP, and FN
For a set detection threshold $\tau$ (e.g. IoU $\ge 0.5$):
*   **True Positive (TP):** A prediction that overlaps a ground-truth box of the same class with $\text{IoU} \ge \tau$. If multiple predictions overlap the same ground-truth, only the one with the highest confidence is a TP; the rest are FPs.
*   **False Positive (FP):** A prediction that has an $\text{IoU} < \tau$ with all ground-truth boxes, or overlaps a ground-truth box of a different class.
*   **False Negative (FN):** A ground-truth box that the model failed to detect with an $\text{IoU} \ge \tau$.

### C. Precision and Recall
*   **Precision ($P$):** The proportion of positive detections that are correct:

$$P = \frac{TP}{TP + FP}$$

*   **Recall ($R$):** The proportion of actual objects that were successfully detected:

$$R = \frac{TP}{TP + FN}$$

### D. Average Precision (AP) and Mean Average Precision (mAP)
Average Precision represents the area under the Precision-Recall curve $p(r)$. In practice, YOLO calculates AP using all-point interpolation:

$$\text{AP} = \sum_{k=1}^N (r_{k+1} - r_k) \max_{\tilde{r} \ge r_{k+1}} p(\tilde{r})$$

Mean Average Precision averages the AP across all $C$ classes:

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c$$

*   **mAP@0.5:** Calculated at a single IoU threshold $\tau = 0.5$. Measures general detection capability.
*   **mAP@0.5:0.95 (COCO mAP):** The average mAP computed over ten thresholds $\tau \in \{0.5, 0.55, 0.60, \dots, 0.95\}$:

$$\text{mAP}_{0.5:0.95} = \frac{1}{10} \sum_{\tau \in \{0.5, 0.55, \dots, 0.95\}} \text{mAP}_{\tau}$$

### E. Non-Maximum Suppression (NMS) Mechanics
During validation and inference, the network outputs multiple overlapping candidate boxes for the same object. NMS filters these down:
1. Sort all predictions by confidence score.
2. Select the box with the highest confidence and save it.
3. Calculate the IoU of this box with all other boxes of the same class.
4. Remove boxes that have an $\text{IoU} \ge \text{iou}_{thresh}$ (suppression).
5. Repeat for remaining boxes.

---

## 3. Comprehensive Parameter Details for `model.val()`

| Parameter | Type | Default Value | Range / Choices | Deep Technical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `data` | `str` | `None` | Path to file | Path to `data.yaml`. Defaults to the config saved in the model checkpoint. |
| `batch` | `int` | `16` | $[1, \infty)$ | Batch size for validation. Set larger for faster validation if VRAM permits. |
| `imgsz` | `int` | `640` | Multiples of 32 | Evaluation image resolution. Must match training size for consistent metrics. |
| `device` | `str/int/list` | `None` | `cpu`, `0`, `[0, 1]` | Device to run validation on. |
| `workers` | `int` | `8` | $[0, \infty)$ | Number of CPU workers to load data. |
| `save_json` | `bool` | `False` | `True`, `False` | Saves validation results to a COCO-formatted JSON file. |
| `conf` | `float` | `0.001` | $(0, 1.0]$ | Confidence threshold. A low default ($0.001$) is used to compute full PR curves. |
| `iou` | `float` | `0.6` | $[0, 1.0)$ | IoU threshold for Non-Maximum Suppression (NMS). |
| `max_det` | `int` | `300` | $[1, \infty)$ | Maximum number of detections allowed per image. |
| `half` | `bool` | `False` | `True`, `False` | Force FP16 half-precision validation, saving VRAM and speed. |
| `rect` | `bool` | `False` | `True`, `False` | Rectangular batching. Groups images of similar aspect ratio together to reduce padding. |

---

## 4. Python vs. CLI Code Comparison

### Python API Usage
```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# Execute validation and collect metrics
metrics = model.val(
    data='datasets/custom_data/data.yaml',
    split='val',
    batch=32,
    imgsz=640,
    device=0
)

# Access primary metrics programmatically
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
```

### CLI Usage
```bash
yolo val \
  model=runs/detect/train/weights/best.pt \
  data=datasets/custom_data/data.yaml \
  split=val \
  batch=32 \
  imgsz=640 \
  device=0
```

---

## 💡 Professor Tips

### 1. Why `conf=0.001` for Validation?
During standard prediction, `conf` is set to `0.25` to ignore noise. However, during validation, `conf` defaults to `0.001`. Why? To calculate the Precision-Recall curve accurately, we need to gather predictions down to very low confidences (otherwise the PR curve would be truncated at $conf=0.25$, preventing correct calculation of Area Under the Curve (AUC) for recall values close to $1.0$).

### 2. Doubling Batch Size for Validation VRAM Efficiency
During training, PyTorch must allocate memory for the computation graph and gradients, requiring significant VRAM. During validation, there is no backpropagation (gradients are disabled under `torch.no_grad()`). As a result, you can typically set `batch` to **double** or **triple** your training batch size without risking an Out-Of-Memory (OOM) error, significantly speeding up evaluation.

---
*Related Topics:*
*   [[EX54_Resuming_Training|EX54: Resuming Training]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
