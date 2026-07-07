# 🧠 EX62: Inference Visualization

After a YOLO model processes an image or video, you need to visualize the predictions. Ultralytics provides built-in utilities like the `plot()` method, which can be easily combined with OpenCV or PIL for custom overlays, drawing, and interactive displays.

---

## 1. Mathematical and Theoretical Foundations of Inference Visualization

Rendering predictions on an image canvas requires mapping coordinates back to the pixel space and overlaying masks/bounding boxes with transparency to preserve background context.

### A. Coordinate Restoration (Normalized to Pixels)
If your model output yields normalized bounding box centroids $(x_c, y_c, w, h)$ on an image of dimensions $W \times H$, they must be converted to top-left and bottom-right pixel integers:

$$x_1 = \text{round}\left( \left( x_c - \frac{w}{2} \right) \cdot W \right), \quad y_1 = \text{round}\left( \left( y_c - \frac{h}{2} \right) \cdot H \right)$$
$$x_2 = \text{round}\left( \left( x_c + \frac{w}{2} \right) \cdot W \right), \quad y_2 = \text{round}\left( \left( y_c + \frac{h}{2} \right) \cdot H \right)$$

### B. Alpha Blending (Semi-Transparent Overlays)
For masks and shaded bounding boxes, overlaying a solid color blocks underlying texture details. We use Alpha Blending to combine a foreground color $\mathbf{C}$ with the background image pixel $\mathbf{I}_{bg}$:

$$\mathbf{I}_{blend}(x,y) = \alpha \mathbf{C} + (1 - \alpha) \mathbf{I}_{bg}(x,y)$$

Where:
*   $\alpha \in [0.0, 1.0]$ represents the opacity (blend coefficient).
*   $\mathbf{C}$ is the target overlay RGB vector.
*   $\mathbf{I}_{bg}(x,y)$ is the original pixel color at coordinates $(x,y)$.

### C. Binary Mask Superposition
For instance segmentation tasks, the model yields a binary mask array $\mathbf{M} \in \{0, 1\}^{H \times W}$. The overlay is applied element-wise:

$$\mathbf{I}_{final}(x, y) = \mathbf{M}(x,y) \cdot \mathbf{I}_{blend}(x,y) + (1 - \mathbf{M}(x,y)) \cdot \mathbf{I}_{bg}(x,y)$$

---

## 2. API Reference: `Results.plot()` Parameters

The `plot()` method on a `Results` object draws detections onto the image array. The table below covers the configuration settings:

| Parameter | Type | Default Value | Range / Choices | Deep Technical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `conf` | `bool` | `True` | `True`, `False` | Render the confidence score float next to the class name label. |
| `line_width` | `int` | `None` | $[1, \infty)$ | Line thickness of bounding boxes. If `None`, dynamically adjusts based on image size. |
| `font_size` | `float` | `None` | $(0, \infty)$ | Font size of text. Scale-adjusted based on dimensions if omitted. |
| `font` | `str` | `'Arial'` | System font name | Font type utilized for text printing. |
| `labels` | `bool` | `True` | `True`, `False` | Toggles drawing the text box containing the class label. |
| `boxes` | `bool` | `True` | `True`, `False` | Toggles drawing the bounding box outline. Can be disabled to show masks only. |
| `probs` | `bool` | `True` | `True`, `False` | Render class probability values (for classification models). |
| `kpt_line` | `bool` | `True` | `True`, `False` | Draw lines connecting joint keypoints (for pose models). |
| `kpt_radius` | `int` | `5` | $[1, \infty)$ | Circle radius in pixels for drawing joints (for pose models). |
| `mask` | `bool` | `True` | `True`, `False` | Toggles rendering semi-transparent instance segmentation masks. |
| `mask_color` | `tuple` | `None` | BGR Color tuple | Hardcodes a single color for all masks instead of generating distinct colors per class. |
| `img` | `ndarray` | `None` | Image array | Overlay predictions on a custom background array rather than the original input frame. |

---

## 3. Python vs. CLI Code Comparison

### Python API Usage
```python
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")
results = model("bus.jpg")

# Render BGR annotated frame
annotated_frame = results[0].plot(conf=True, line_width=2)

# Display using OpenCV
cv2.imshow("Annotated Frame", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### CLI Usage
```bash
# Predict and save visualization to runs/detect/predict/
yolo detect predict model=yolo11n.pt source="bus.jpg" save=True
```

---

## 💡 Professor Tips

### 1. Headless Server Environment Crashes
If you run `cv2.imshow` or `results[0].plot(show=True)` on a headless remote server (e.g., AWS EC2, Google Cloud, Docker container) without an active X11 display server, your code will crash:
`cv2.error: OpenCV(4.x) [x11] ... Cannot connect to X server`

**Remediation:**
1.  Verify the environment has no GUI display, and suppress drawing windows.
2.  Set `show=False` and use `save=True` to write outputs to disk.
3.  Alternatively, compile OpenCV with headless support: `pip install opencv-python-headless`.

### 2. BGR vs. RGB Color Channels
OpenCV stores images in BGR format, while Matplotlib and PIL process images in RGB format. The `Results.plot()` method outputs BGR arrays by default. To display a plot in Matplotlib, you must invert the channels:
```python
import matplotlib.pyplot as plt

# Convert from BGR to RGB
rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_frame)
plt.show()
```

---
*Related Topics:*
*   [[EX57_Bounding_Box_Parsing|EX57: Bounding Box Parsing]]
*   [[EX60_Streaming_Inference|EX60: Streaming Inference]]
*   [[EX63_Multi_Object_Tracking|EX63: Multi-Object Tracking (MOT)]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
