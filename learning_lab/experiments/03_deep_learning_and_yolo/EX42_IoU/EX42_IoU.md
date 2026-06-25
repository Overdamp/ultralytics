# 🧠 EX42: Intersection over Union (IoU)

Intersection over Union (IoU) is a metric that measures the overlap between two bounding boxes: the predicted bounding box ($A$) and the ground truth bounding box ($B$).

---

## 1. Mathematical Formulation

Given two boxes in XYXY format:
*   Box A: $[x_1^A, y_1^A, x_2^A, y_2^A]$
*   Box B: $[x_1^B, y_1^B, x_2^B, y_2^B]$

### Step 1: Area of Intersection
The intersection is the overlapping rectangular area. We calculate its coordinates by taking the maximum of the top-left corners and the minimum of the bottom-right corners:

$$x_{\min}^{\text{inter}} = \max(x_1^A, x_1^B), \quad y_{\min}^{\text{inter}} = \max(y_1^A, y_1^B)$$
$$x_{\max}^{\text{inter}} = \min(x_2^A, x_2^B), \quad y_{\max}^{\text{inter}} = \min(y_2^A, y_2^B)$$

The width ($w$) and height ($h$) of the intersection are:
$$w_{\text{inter}} = \max(0, x_{\max}^{\text{inter}} - x_{\min}^{\text{inter}})$$
$$h_{\text{inter}} = \max(0, y_{\max}^{\text{inter}} - y_{\min}^{\text{inter}})$$

$$\text{Area}_{\text{inter}} = w_{\text{inter}} \times h_{\text{inter}}$$

### Step 2: Area of Union
The union area combines the areas of both boxes minus the overlapping intersection area (to avoid double counting):

$$\text{Area}_A = (x_2^A - x_1^A) \times (y_2^A - y_1^A)$$
$$\text{Area}_B = (x_2^B - x_1^B) \times (y_2^B - y_1^B)$$
$$\text{Area}_{\text{union}} = \text{Area}_A + \text{Area}_B - \text{Area}_{\text{inter}}$$

### Step 3: Compute IoU
$$\text{IoU} = \frac{\text{Area}_{\text{inter}}}{\text{Area}_{\text{union}}}$$

---

## 2. Advanced IoU Variants (YOLO Loss Functions)
Standard IoU has a major flaw: if two boxes do not overlap, their intersection is $0$, meaning $\text{IoU} = 0$. Since the loss becomes a constant flat surface, gradient descent cannot update the box coordinates.

To solve this, advanced versions were introduced:
*   **GIoU (Generalized IoU):** Adds a penalty term based on the area of the smallest enclosing convex box $C$ containing both $A$ and $B$, ensuring gradients flow even when there is no overlap.
*   **DIoU (Distance IoU):** Penalizes the normalized distance between the center points of the two boxes, making boxes converge much faster.
*   **CIoU (Complete IoU - used in YOLO):** Adds an aspect ratio penalty term, measuring three key geometric factors: overlap area, center point distance, and aspect ratio consistency.

---

## 💻 Python Implementation (NumPy)

```python
import numpy as np

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    x1_inter = max(boxA[0], boxB[0])
    y1_inter = max(boxA[1], boxB[1])
    x2_inter = min(boxA[2], boxB[2])
    y2_inter = min(boxA[3], boxB[3])
    
    # Calculate intersection area
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    
    # Calculate union area
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_union = areaA + areaB - area_inter
    
    # Compute IoU
    if area_union == 0:
        return 0.0
    return area_inter / area_union

# Example usage:
box_pred = [100, 100, 210, 210]
box_truth = [120, 120, 220, 220]
print(f"IoU: {calculate_iou(box_pred, box_truth):.4f}")
```

---
*Related Topics:*
*   [[EX41_Bounding_Box\|EX41: Bounding Box Coordinates]]
*   [[EX43_NMS\|EX43: Non-Maximum Suppression (NMS)]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
