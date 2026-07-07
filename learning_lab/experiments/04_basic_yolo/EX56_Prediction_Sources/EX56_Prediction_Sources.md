# 🧠 EX56: YOLO Prediction Sources

In real-world computer vision applications, model inference is rarely limited to simple static image files. An end-to-end system must ingest data from diverse streams: static local files, video file buffers, raw memory arrays, real-time cameras, or network streams. 

Ultralytics YOLO abstracts this complexity through a unified input processor within `model.predict()`. Under the hood, YOLO inspects the input type and instantiates the appropriate data loader, automatically handling decoding, resizing, normalization, and tensor conversion.

---

## 1. Mathematical and Theoretical Foundations of Inference Preprocessing

Before passing an image through the network, the input source must be transformed into a standardized mathematical tensor format. This pipeline consists of scaling, color space alignment, and dimension permuting.

### A. Affine Bounding Box Mapping (Letterboxing Coordinate Restoration)
Letterboxing scales the input image while preserving the aspect ratio. When predicting, the bounding box coordinates are predicted in the scaled network resolution coordinates $(x_{net}, y_{net})$. To map them back to the original image coordinates $(x_{orig}, y_{orig})$, we apply the inverse affine transformation:

Given scaling factor $s$ and padding offsets $(t_x, t_y)$:

$$s = \min\left(\frac{W_{net}}{W_{orig}}, \frac{H_{net}}{H_{orig}}\right)$$
$$t_x = \frac{W_{net} - (W_{orig} \times s)}{2}, \quad t_y = \frac{H_{net} - (H_{orig} \times s)}{2}$$

The forward transformation from original image to network input is:

$$\begin{pmatrix} x_{net} \\ y_{net} \end{pmatrix} = s \begin{pmatrix} x_{orig} \\ y_{orig} \end{pmatrix} + \begin{pmatrix} t_x \\ t_y \end{pmatrix}$$

Therefore, the inverse transformation to retrieve original bounding box coordinates is:

$$x_{orig} = \frac{x_{net} - t_x}{s}, \quad y_{orig} = \frac{y_{net} - t_y}{s}$$

### B. Color Space Transformations and Channel Ordering
*   **BGR vs. RGB:** OpenCV (`cv2`) loads images in BGR format by default. PyTorch models expect RGB channel ordering. 
*   **Data Layout Conversion:** OpenCV represents images as HWC arrays (Height, Width, Channel). Convolutional layers in PyTorch operate on BCHW layout (Batch, Channel, Height, Width).
*   **Mathematical scaling:** Integer pixel values $[0, 255]$ are divided by $255.0$ to produce floating-point inputs in the range $[0.0, 1.0]$.

$$\mathbf{X}_{tensor} = \text{Permute}\left(\frac{\mathbf{X}_{array}}{255.0}\right) \in \mathbb{R}^{3 \times H_{net} \times W_{net}}$$

---

## 2. Comprehensive Parameter Details for `model.predict()`

The inference engine in YOLO features several settings for modifying input processing and output formatting:

| Parameter | Type | Default Value | Range / Choices | Deep Technical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `source` | `str/int/array` | *Required* | Path, URL, array, or device ID | The input source. Supports paths to images/directories, video files, RTSP/HTTP streams, YouTube URLs, webcams (`0`), numpy arrays, or PIL images. |
| `conf` | `float` | `0.25` | $[0.0, 1.0]$ | Confidence score threshold. Detections below this value are filtered out immediately. |
| `iou` | `float` | `0.7` | $[0.0, 1.0)$ | IoU threshold for Non-Maximum Suppression (NMS). Controls how overlapping boxes are merged. |
| `imgsz` | `int/tuple` | `640` | Multiples of 32 | Target network resolution. Can be set as `(height, width)` for rectangular shapes. |
| `device` | `str/int` | `None` | `cpu`, `0`, `cuda` | Target hardware. |
| `half` | `bool` | `False` | `True`, `False` | Enables FP16 half-precision inference, doubling speed and saving VRAM on supported GPUs. |
| `save` | `bool` | `False` | `True`, `False` | Saves annotated images/videos to the run directory. |
| `save_txt` | `bool` | `False` | `True`, `False` | Saves prediction bounding boxes to text files in YOLO format (`class x_center y_center width height`). |
| `save_conf` | `bool` | `False` | `True`, `False` | Saves class confidence scores alongside box coordinates in text outputs. |
| `save_crop` | `bool` | `False` | `True`, `False` | Crops detected objects and saves them as separate images. |
| `show` | `bool` | `False` | `True`, `False` | Displays inference results in real-time using an OpenCV window. |
| `vid_stride` | `int` | `1` | $[1, \infty)$ | Frame stride for processing videos. E.g., `vid_stride=2` skips every other frame to save compute. |
| `stream_buffer`| `bool` | `False` | `True`, `False` | Buffers incoming frames for streams. If `False`, drops older frames to maintain real-time low latency. |
| `visualize` | `bool` | `False` | `True`, `False` | Visualizes intermediate feature maps of the model layers. |
| `augment` | `bool` | `False` | `True`, `False` | Enables Test-Time Augmentation (TTA) to increase accuracy at the cost of slower inference. |
| `agnostic_nms` | `bool` | `False` | `True`, `False` | Performs class-agnostic NMS. Prevents overlapping boxes of *different* classes. |
| `classes` | `int/list` | `None` | Class IDs | Filter results to only show specified class indices (e.g. `classes=[0, 2]`). |

---

## 3. Python vs. CLI Code Comparison

### Python API Usage
```python
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Inference on PIL Image
pil_img = Image.open("demo.jpg")
results_pil = model.predict(source=pil_img)

# Inference on OpenCV Image (BGR NumPy array)
numpy_img = cv2.imread("demo.jpg")
results_cv2 = model.predict(source=numpy_img)

# Inference on RTSP network stream (using stream=True to return a generator)
results_stream = model.predict(source="rtsp://admin:pass@192.168.1.100/stream1", stream=True)
for r in results_stream:
    # Process generator frames dynamically to save system memory
    boxes = r.boxes
```

### CLI Usage
```bash
# Ingest local file
yolo detect predict model=yolo11n.pt source="demo.jpg" save=True

# Ingest webcam stream
yolo detect predict model=yolo11n.pt source=0 show=True

# Ingest stream with video striding
yolo detect predict model=yolo11n.pt source="traffic.mp4" vid_stride=5 save=True
```

### Key Differences Comparison
*   **RAM Footprint:** In Python, passing `stream=True` allows processing infinite video streams by returning a generator that yields one frame's results at a time, keeping RAM usage low. In contrast, running without `stream=True` stores all frames in memory, causing OOM issues. The CLI automatically handles large videos efficiently but does not yield objects for code pipelines.
*   **Deployment Integration:** In Python, passing direct memory objects (NumPy/PIL) avoids file system write/read penalties, which is required when deploying models behind web APIs (e.g., FastAPI).

---

## 💡 Professor Tips

### Low-Memory Generator Patterns
Always use `stream=True` when running inference on long video files or continuous camera feeds:
```python
# GOOD: Low RAM usage (keeps only current frame in memory)
for result in model.predict(source="video.mp4", stream=True):
    pass

# BAD: High RAM usage (accumulates all results in a list)
results = model.predict(source="video.mp4")
```

### Thread Safety in Web APIs
If you deploy YOLO inside multi-threaded servers (like FastAPI with `async def` or Flask), instantiate the model outside the request loop. Note that CUDA operations are thread-bound; it is often safer to run model inference on the CPU or enforce single-thread access using thread locks to prevent PyTorch CUDA initialization errors.

---
*Related Topics:*
*   [[EX57_Bounding_Box_Parsing|EX57: Bounding Box Parsing]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
