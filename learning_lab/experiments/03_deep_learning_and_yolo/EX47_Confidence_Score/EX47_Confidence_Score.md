# 🧠 EX47: Confidence Score in Object Detection

The confidence score is a probability metric indicating how likely a predicted bounding box contains an object and how accurately the box fits that object.

---

## 1. Mathematical Formulation

Across different generations of YOLO, the computation of the confidence score has evolved:

### A. Classic YOLO (v1 - v5)
In classic anchor-based architectures, the confidence score for a bounding box is formulated as:

$$\text{Confidence} = P(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$$

Where:
*   $P(\text{Object}) \in [0, 1]$ is the **objectness score** (the probability that the grid cell contains *any* object at all).
*   $\text{IoU}_{\text{pred}}^{\text{truth}}$ is the overlap between the predicted bounding box and the true bounding box.

The final class-specific confidence score is:
$$\text{Class Score} = P(\text{Class}_i | \text{Object}) \times \text{Confidence} = P(\text{Class}_i | \text{Object}) \times P(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$$

### B. Anchor-Free YOLO (v8 - v11)
Modern anchor-free YOLO architectures remove the separate objectness branch to decrease latency. Instead, they directly output class probabilities, and the confidence score is calculated during optimization using a **Task-Aligned Assigner**, which combines classification and localization quality into a single score:

$$t = s^\alpha \times \text{IoU}^\beta$$

Where:
*   $s$ is the predicted class probability.
*   $\text{IoU}$ is the overlap between prediction and ground-truth.
*   $\alpha$ and $\beta$ are weighting factors.

---

## 🔬 Computer Vision Case Study: Tuning the `conf` Parameter
When running inference on your PTT dataset, adjusting the confidence threshold (`conf`) has a huge impact on your model's real-world usability.

### Scenario: Running Inference with different thresholds

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train-3/weights/best.pt')

# Scenario A: High confidence threshold (High Precision, Low Recall)
# Only predicts objects it is 60% sure about. Excellent for avoiding false alarms.
# May miss smaller valves (e.g. 'small-valve').
results_a = model('datasets/coco8/overall-ptt-object-detection.v11i.yolov11/test/images', conf=0.6)

# Scenario B: Low confidence threshold (Low Precision, High Recall)
# Predicts everything down to 15% certainty. Good for picking up every object.
# Will introduce false positives (clutter labeled as 'flange' or 'valve').
results_b = model('datasets/coco8/overall-ptt-object-detection.v11i.yolov11/test/images', conf=0.15)
```

---
*Related Topics:*
*   [[EX46_YOLO_Grid\|EX46: YOLO Grid System]]
*   [[EX48_mAP\|EX48: Mean Average Precision (mAP)]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
