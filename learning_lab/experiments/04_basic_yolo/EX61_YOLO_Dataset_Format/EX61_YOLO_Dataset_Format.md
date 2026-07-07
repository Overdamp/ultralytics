# 🧠 EX61: YOLO Dataset Format

To train an Ultralytics YOLO model, your data must be structured in a specific directory layout and annotated using normalized coordinates. Understanding this format is essential for preparing custom datasets and preventing training-time alignment errors.

---

## 1. Mathematical and Theoretical Foundations of Normalized Annotations

Object detection datasets use different coordinate conventions. YOLO uses normalized bounding boxes to make annotations independent of input resolutions, supporting dynamic multi-scale training.

### A. Coordinate System Conversions

#### 1. Conversion from Pascal VOC ($[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ absolute pixels) to YOLO ($[x_c, y_c, w, h]$ normalized):
Given image dimensions width $W$ and height $H$:

$$w_{pixel} = x_{\max} - x_{\min}$$
$$h_{pixel} = y_{\max} - y_{\min}$$

Normalization scales coordinates relative to the image dimensions:

$$x_c = \frac{x_{\min} + \frac{w_{pixel}}{2}}{W} = \frac{x_{\min} + x_{\max}}{2W}$$
$$y_c = \frac{y_{\min} + \frac{h_{pixel}}{2}}{H} = \frac{y_{\min} + y_{\max}}{2H}$$
$$w = \frac{w_{pixel}}{W} = \frac{x_{\max} - x_{\min}}{W}$$
$$h = \frac{h_{pixel}}{H} = \frac{y_{\max} - y_{\min}}{H}$$

#### 2. Conversion from COCO ($[x_{\min}, y_{\min}, w_{pixel}, h_{pixel}]$ absolute pixels) to YOLO ($[x_c, y_c, w, h]$ normalized):

$$x_c = \frac{x_{\min} + \frac{w_{pixel}}{2}}{W}$$
$$y_c = \frac{y_{\min} + \frac{h_{pixel}}{2}}{H}$$
$$w = \frac{w_{pixel}}{W}$$
$$h = \frac{h_{pixel}}{H}$$

### B. Why Coordinate Normalization Matters
If bounding boxes were saved in absolute pixels, scaling input resolutions dynamically during multi-scale training would require recalculating target labels on every batch. By keeping coordinates in the normalized range $[0.0, 1.0]$, coordinates remain constant. The network multiplies the normalized values by the current batch width and height:

$$x_{net\_pixel} = x_c \cdot W_{net}$$

---

## 2. Dataset Annotation Formatting Schemas

Different computer vision tasks require distinct annotation schemas within the label `.txt` files:

### 1. Object Detection Annotation Format
Each line corresponds to a single bounding box:
```text
class_id x_center y_center width height
```
*Example (Class 0 at image center, size $50\%$ width and $30\%$ height):*
`0 0.500000 0.500000 0.500000 0.300000`

### 2. Instance Segmentation Annotation Format
Each line defines an object polygon boundary:
```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```
*Example (Class 1 defined by a 3-point triangle boundary):*
`1 0.100000 0.100000 0.200000 0.400000 0.300000 0.100000`

### 3. Pose Estimation Annotation Format
Each line contains a bounding box and joint keypoint coordinates:
```text
class_id x_center y_center width height px1 py1 pv1 px2 py2 pv2 ... pxn pyn pvn
```
*   Where $(px_i, py_i)$ are normalized keypoint coordinates, and $pv_i$ is the visibility flag ($0$: not labeled, $1$: labeled but occluded, $2$: labeled and visible).

---

## 3. Directory Layout and `data.yaml` Reference

YOLO requires a strict directory layout separating images from annotations:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/  (img01.jpg)
│   └── labels/  (img01.txt)
└── val/
    ├── images/  (img02.jpg)
    └── labels/  (img02.txt)
```

### The `data.yaml` Configurations
| Parameter | Type | Required | Range / Choices | Deep Technical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `path` | `str` | Yes | Path string | The parent directory of the dataset. Can be relative or absolute. |
| `train` | `str/list` | Yes | Relative paths | Path to training images folder (e.g. `train/images`). |
| `val` | `str/list` | Yes | Relative paths | Path to validation images folder. |
| `test` | `str` | No | Relative path | Path to test images folder (optional). |
| `nc` | `int` | Yes | $[1, \infty)$ | Number of classes defined in the dataset. |
| `names` | `dict/list`| Yes | Index mapping | Mapping of class IDs to human readable labels. |

---

## 4. Python Code Helper: Conversion and Validation

### Python Helper Script: Converting Pascal VOC to YOLO Format
```python
import cv2
import numpy as np

def voc_to_yolo(img_path, voc_box):
    """
    Converts Pascal VOC box [xmin, ymin, xmax, ymax] to normalized YOLO box.
    """
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    
    xmin, ymin, xmax, ymax = voc_box
    
    # Clip coordinates to prevent out-of-bounds bounding boxes
    xmin = np.clip(xmin, 0, W - 1)
    xmax = np.clip(xmax, 0, W - 1)
    ymin = np.clip(ymin, 0, H - 1)
    ymax = np.clip(ymax, 0, H - 1)
    
    w_pixel = xmax - xmin
    h_pixel = ymax - ymin
    
    xc = (xmin + w_pixel / 2) / W
    yc = (ymin + h_pixel / 2) / H
    w = w_pixel / W
    h = h_pixel / H
    
    return [xc, yc, w, h]
```

---

## 💡 Professor Tips

### 1. Including Negative Samples (Background Images)
If your model false-detects background noise (e.g., predicting valves on empty pipes), add background images that contain no targets. For these images, add the image to the `images/` directory and create an **empty** label file in `labels/` with size 0 bytes. This teaches the model to identify empty backgrounds, lowering the false-positive rate.

### 2. Class Index Alignment
YOLO class IDs are zero-indexed: $[0, 1, \dots, nc - 1]$. If your annotations are one-indexed (e.g., from export sources starting at $1$), you must subtract 1 from all class IDs. Failing to do so will shift class mappings or trigger index-out-of-range errors during loss computation.

---
*Related Topics:*
*   [[EX62_Inference_Visualization|EX62: Inference Visualization]]
*   [[EX53_Training_Settings|EX53: Training Settings]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
