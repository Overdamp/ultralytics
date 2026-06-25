# 🧠 EX41: Bounding Box Coordinate Systems

Bounding boxes define the spatial location and dimensions of objects in an image. Understanding how to represent, convert, and normalize these coordinates is essential for handling object detection pipelines.

---

## 1. Coordinate Representation Formats

There are two primary bounding box formats used in computer vision:

### A. Corners Format (XYXY)
*   **Representation:** $[x_1, y_1, x_2, y_2]$
*   **Description:** Coordinates of the top-left corner $(x_1, y_1)$ and bottom-right corner $(x_2, y_2)$.
*   **Use Case:** Default format for plotting (OpenCV bounding boxes) and computing overlap metrics like IoU.

### B. Centroid Format (XYWH)
*   **Representation:** $[x_c, y_c, w, h]$
*   **Description:** Coordinates of the box center $(x_c, y_c)$ along with the width ($w$) and height ($h$) of the box.
*   **Use Case:** Primary format predicted by neural network heads.

---

## 2. Bounding Box Conversion Math

### Converting XYWH to XYXY:
Given a box $[x_c, y_c, w, h]$:

$$x_1 = x_c - \frac{w}{2}, \quad y_1 = y_c - \frac{h}{2}$$
$$x_2 = x_c + \frac{w}{2}, \quad y_2 = y_c + \frac{h}{2}$$

### Converting XYXY to XYWH:
Given a box $[x_1, y_1, x_2, y_2]$:

$$x_c = \frac{x_1 + x_2}{2}, \quad y_c = \frac{y_1 + y_2}{2}$$
$$w = x_2 - x_1, \quad h = y_2 - y_1$$

---

## 3. YOLO Normalized Annotation Format
To ensure that bounding box labels are **scale-invariant** (usable on images of different sizes without breaking coordinates), YOLO stores annotations normalized between `0` and `1`:

Given an image of width $W$ and height $H$, coordinates are normalized as:

$$x_{\text{norm}} = \frac{x_c}{W}, \quad y_{\text{norm}} = \frac{y_c}{H}$$
$$w_{\text{norm}} = \frac{w}{W}, \quad h_{\text{norm}} = \frac{h}{H}$$

In the text file label corresponding to each image, YOLO expects:
```text
<class_id> <x_norm> <y_norm> <w_norm> <h_norm>
```

---

## 💻 Python Coordinate Conversion Functions (NumPy)

```python
import numpy as np

def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from [xc, yc, w, h] to [x1, y1, x2, y2].
    Can handle single box list or a NumPy array of shape (N, 4).
    """
    boxes = np.array(boxes)
    converted = np.zeros_like(boxes)
    converted[..., 0] = boxes[..., 0] - (boxes[..., 2] / 2) # x1
    converted[..., 1] = boxes[..., 1] - (boxes[..., 3] / 2) # y1
    converted[..., 2] = boxes[..., 0] + (boxes[..., 2] / 2) # x2
    converted[..., 3] = boxes[..., 1] + (boxes[..., 3] / 2) # y2
    return converted

def xyxy_to_xywh(boxes):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [xc, yc, w, h].
    """
    boxes = np.array(boxes)
    converted = np.zeros_like(boxes)
    converted[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2 # xc
    converted[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2 # yc
    converted[..., 2] = boxes[..., 2] - boxes[..., 0]       # w
    converted[..., 3] = boxes[..., 3] - boxes[..., 1]       # h
    return converted
```

---
*Related Topics:*
*   [[EX40_Embeddings\|EX40: Feature Embeddings]]
*   [[EX42_IoU\|EX42: Intersection over Union (IoU)]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
