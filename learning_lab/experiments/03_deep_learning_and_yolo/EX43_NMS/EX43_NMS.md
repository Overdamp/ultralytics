# 🧠 EX43: Non-Maximum Suppression (NMS)

Non-Maximum Suppression (NMS) is a post-processing algorithm used in object detection to eliminate redundant, overlapping bounding boxes predicted for the same object, leaving only the single best prediction.

---

## 1. Why NMS is Necessary
Single-stage object detectors (like YOLO) evaluate thousands of grid cells across different scales. Often, multiple cells surrounding an object will all predict high confidence scores for the same item, resulting in multiple overlapping boxes. NMS filters these out to ensure each object has exactly one detection box.

---

## 2. The NMS Algorithm

### Inputs:
*   $B = \{b_1, \dots, b_n\}$: Candidate bounding boxes.
*   $S = \{s_1, \dots, s_n\}$: Corresponding prediction confidence scores.
*   $N_t$: The IoU threshold (typically `0.45` to `0.7`).

### Steps:
1.  Initialize an empty list for selected detections $D = \{\}$.
2.  While $B$ is not empty:
    *   Find the box $M$ with the highest confidence score in $S$.
    *   Remove $M$ from $B$ and add it to $D$.
    *   For every remaining box $b_i$ in $B$, compute the overlap $\text{IoU}(M, b_i)$.
    *   If $\text{IoU}(M, b_i) \ge N_t$, discard $b_i$ and its score $s_i$ from $B$ and $S$.
3.  Return the final detection list $D$.

---

## 3. Advanced Variant: Soft-NMS
In standard (hard) NMS, if an actual second object overlaps with the primary object (e.g., two valves placed side-by-side on a manifold), hard NMS will completely delete the second box, causing a **False Negative**.

**Soft-NMS** solves this by decaying the confidence score of overlapping boxes instead of deleting them. If the decayed score is still above the confidence threshold, the box is preserved:

$$s_i \leftarrow s_i \cdot \exp\left( -\frac{\text{IoU}(M, b_i)^2}{\sigma} \right)$$

---

## 💻 Python Implementation (NumPy)

Here is a fast NumPy implementation of 1D coordinate NMS:

```python
import numpy as np

def nms_numpy(boxes, scores, iou_threshold):
    # Coordinates of boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort boxes by confidence score descending
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate overlap coordinates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Calculate overlap area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        # Compute IoU
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Keep boxes where IoU is less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep
```

---
*Related Topics:*
*   [[EX42_IoU\|EX42: Intersection over Union (IoU)]]
*   [[EX44_Object_Detection_Pipeline\|EX44: Object Detection Pipeline]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
