# 🧠 EX57: Bounding Box Parsing

In real-world applications, running inference is only the first step. To make decisions (such as triggering an alarm, counting objects, or cropping a region of interest), you must programmatically inspect and extract coordinates, class labels, and confidence scores from YOLO's predictions.

Ultralytics YOLO simplifies this by returning a list of `Results` objects. This lesson explains how to traverse and parse this data structure using the Python API.

---

## 1. Bounding Box Coordinate System Transformations

In computer vision, bounding boxes are represented in different coordinate systems depending on the target task.

### A. Coordinate Formats and Math

#### 1. XYXY (Absolute Pixel Coordinates)
Defined by the top-left corner $(x_1, y_1)$ and bottom-right corner $(x_2, y_2)$ in pixels:

$$\mathbf{B}_{xyxy} = [x_1, y_1, x_2, y_2]$$

#### 2. XYWH (Absolute Centroid Coordinates)
Defined by the bounding box center $(x_c, y_c)$, width $w$, and height $h$ in pixels:

$$\mathbf{B}_{xywh} = [x_c, y_c, w, h]$$

*Conversion from XYXY to XYWH:*

$$x_c = \frac{x_1 + x_2}{2}, \quad y_c = \frac{y_1 + y_2}{2}$$
$$w = x_2 - x_1, \quad h = y_2 - y_1$$

*Conversion from XYWH to XYXY:*

$$x_1 = x_c - \frac{w}{2}, \quad y_1 = y_c - \frac{h}{2}$$
$$x_2 = x_c + \frac{w}{2}, \quad y_2 = y_c + \frac{h}{2}$$

#### 3. Normalized Coordinate Formats (XYXYN and XYWHN)
To make coordinates scale-invariant across different image resolutions, coordinates are divided by the original image dimensions $W$ and $H$:

$$\mathbf{B}_{xyxyn} = \left[ \frac{x_1}{W}, \frac{y_1}{H}, \frac{x_2}{W}, \frac{y_2}{H} \right]$$
$$\mathbf{B}_{xywhn} = \left[ \frac{x_c}{W}, \frac{y_c}{H}, \frac{w}{W}, \frac{h}{H} \right]$$

---

## 2. API Reference: The `Results` and `Boxes` Objects

When `model(source)` is executed, it returns a `list` of `Results` objects. Here are the key attributes and methods:

### The `Results` Object Structure
| Attribute/Method | Return Type | Deep Technical Explanation |
| :--- | :--- | :--- |
| `.boxes` | `Boxes` | Bounding box coordinates, confidences, and class IDs. |
| `.masks` | `Masks` | Segmentation masks (returns `None` for pure detection models). |
| `.keypoints` | `Keypoints` | Keypoint coordinates for pose estimation. |
| `.probs` | `Probs` | Classification probabilities (for classification tasks). |
| `.orig_img` | `numpy.ndarray` | The original input image array ($H \times W \times C$ in BGR). |
| `.orig_shape` | `tuple` | The original dimensions of the input image `(height, width)`. |
| `.names` | `dict` | Index-to-string dictionary mapping for classes (e.g. `{0: 'person'}`). |
| `.tojson()` | `str` | Serializes the detection metrics into a JSON formatted string. |
| `.plot()` | `numpy.ndarray` | Returns an image with visual boxes and labels drawn. |

### The `Boxes` Object Structure
| Attribute | Return Type | Deep Technical Explanation |
| :--- | :--- | :--- |
| `.xyxy` | `torch.Tensor` | Bounding boxes in $[x_1, y_1, x_2, y_2]$ absolute pixel formats (shape: $N \times 4$). |
| `.xywh` | `torch.Tensor` | Bounding boxes in $[x_c, y_c, w, h]$ absolute pixel formats (shape: $N \times 4$). |
| `.xyxyn` | `torch.Tensor` | Normalized bounding boxes in $[x_1, y_1, x_2, y_2]$ (shape: $N \times 4$). |
| `.xywhn` | `torch.Tensor` | Normalized bounding boxes in $[x_c, y_c, w, h]$ (shape: $N \times 4$). |
| `.conf` | `torch.Tensor` | Confidence scores of detections (shape: $N$). |
| `.cls` | `torch.Tensor` | Integer class ID predictions (shape: $N$). |
| `.id` | `torch.Tensor` | Unique object tracking IDs (shape: $N$, returns `None` if tracking is disabled). |

---

## 3. Python API Parsing Demonstrations

### Vectorized Parsing vs. Iterative Parsing
Iterating through PyTorch tensors box-by-box is highly inefficient. Instead, move tensors to the host memory in one step using vectorized operations.

```python
import cv2
import numpy as np
from ultralytics import YOLO

# Load model and run inference
model = YOLO("yolo11n.pt")
results = model("demo.jpg")
first_result = results[0]

# Verify if objects are detected
if first_result.boxes is not None and len(first_result.boxes) > 0:
    boxes = first_result.boxes
    
    # VECTORIZED: Fast, converts entire tensors at once
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy().astype(float)
    xyxy_coords = boxes.xyxy.cpu().numpy()
    
    # Process using efficient zip structure
    for idx, (cls_id, conf, coords) in enumerate(zip(cls_ids, confs, xyxy_coords)):
        label = first_result.names[cls_id]
        print(f"[{idx}] {label}: Conf={conf:.2f}, Box={coords}")
```

### CLI Command Equivalence (Data Serialization)
Since the CLI cannot return Python objects, you must write predictions to files or export JSON formats:
```bash
# Save predictions directly to a .txt annotation folder
yolo detect predict model=yolo11n.pt source="demo.jpg" save_txt=True save_conf=True

# Run inference and print output in serializable JSON text structure to stdout
yolo detect predict model=yolo11n.pt source="demo.jpg" save_json=True
```

---

## 💡 Professor Tips

### 1. Vectorized Memory Transfer
Never perform CPU conversions inside a detection loop:
```python
# BAD (forces transfer CPU overhead inside every iteration)
for box in boxes:
    coord = box.xyxy[0].cpu().numpy()

# GOOD (transfers the entire tensor block once)
coords = boxes.xyxy.cpu().numpy()
for coord in coords:
    pass
```

### 2. Bounding Box Boundary Clipping
When cropping Regions of Interest (ROI) using coordinates, predicted coordinates can occasionally exceed image dimensions due to fractional rounding or model predictions near the border. Always clip coordinates to the image boundaries to avoid out-of-bounds array crashes:

$$x_1 = \max(0, \min(\text{round}(x_1), W_{orig} - 1))$$
$$y_1 = \max(0, \min(\text{round}(y_1), H_{orig} - 1))$$
$$x_2 = \max(0, \min(\text{round}(x_2), W_{orig} - 1))$$
$$y_2 = \max(0, \min(\text{round}(y_2), H_{orig} - 1))$$

```python
img = first_result.orig_img
H, W, _ = img.shape

for coord in xyxy_coords:
    x1 = int(np.clip(round(coord[0]), 0, W - 1))
    y1 = int(np.clip(round(coord[1]), 0, H - 1))
    x2 = int(np.clip(round(coord[2]), 0, W - 1))
    y2 = int(np.clip(round(coord[3]), 0, H - 1))
    
    crop = img[y1:y2, x1:x2]
    # Perform downstream classification or save crop
```

---
*Related Topics:*
*   [[EX56_Prediction_Sources|EX56: Prediction Sources]]
*   [[EX58_Model_Export|EX58: Model Export & Optimization]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
