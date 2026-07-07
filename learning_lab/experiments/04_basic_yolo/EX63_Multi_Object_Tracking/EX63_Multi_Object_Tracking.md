# 🧠 EX63: Multi-Object Tracking (MOT)

While standard object detection locates objects frame-by-frame, it does not maintain their identities over time. Multi-Object Tracking (MOT) resolves this by assigning a unique, persistent ID to each detected object and tracking its trajectory across successive video frames.

---

## 1. Mathematical and Theoretical Foundations of Multi-Object Tracking

Multi-Object Tracking matches detections across time by modeling object dynamics and appearance profiles.

### A. Kalman Filter State Representation
YOLO trackers utilize a Kalman Filter to predict object states. The tracking state vector $\mathbf{x}_t$ represents the bounding box coordinates and their first-order derivatives (velocities):

$$\mathbf{x}_t = [x_c, y_c, a, h, \dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}]^T$$

Where:
*   $(x_c, y_c)$ represents the absolute center pixel coordinates.
*   $a = \frac{w}{h}$ represents the box aspect ratio.
*   $h$ is the box height.
*   $\dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}$ are the corresponding frame-to-frame change velocities.

The linear state transition model predicts the next state:

$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{z}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t$$

Where:
*   $\mathbf{F}$ is the state transition matrix.
*   $\mathbf{H}$ is the measurement mapping matrix.
*   $\mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q})$ and $\mathbf{v}_t \sim \mathcal{N}(0, \mathbf{R})$ represent the process and measurement Gaussian noise vectors.

### B. Bipartite Matching and Hungarian Algorithm
At each frame $t$, predicted tracker bounding boxes $\mathbf{P}_i$ must be associated with new detections $\mathbf{D}_j$. We compute a cost matrix $\mathbf{C} \in \mathbb{R}^{M \times N}$ based on distance metrics:

$$\mathbf{C}_{i,j} = w_m \cdot (1 - \text{IoU}(\mathbf{P}_i, \mathbf{D}_j)) + (1 - w_m) \cdot d_{\text{cosine}}(\mathbf{f}_i, \mathbf{f}_j)$$

Where:
*   $\text{IoU}$ is the Intersection over Union overlap.
*   $d_{\text{cosine}}$ is the cosine distance between visual appearance embedding vectors $\mathbf{f}$ extracted from a Re-Identification (Re-ID) neural network.
*   $w_m$ is a weighting coefficient balancing motion overlap and visual similarity.

The Hungarian (Kuhn-Munkres) algorithm solves the linear assignment problem to minimize the total matching cost:

$$\min \sum_{i=1}^M \sum_{j=1}^N \mathbf{C}_{i,j} \mathbf{X}_{i,j}$$

Subject to $\mathbf{X}_{i,j} \in \{0, 1\}$ being a binary matching matrix where each detection maps to at most one tracker.

---

## 2. Tracking Algorithms: BoT-SORT vs. ByteTrack

YOLO provides two state-of-the-art trackers pre-configured out of the box:

| Feature | BoT-SORT (`botsort.yaml`) | ByteTrack (`bytetrack.yaml`) |
| :--- | :--- | :--- |
| **Motion Model** | Kalman Filter + Camera Motion Compensation (CMC) | Kalman Filter |
| **Appearance Re-ID**| Yes (incorporates visual features) | No (strictly IoU + Kalman motion) |
| **Inference Speed** | Moderate (visual embedding extraction overhead) | High (fast and lightweight) |
| **Primary Strength** | Robust against camera motion (panning/zooming) | Excellent for static camera views with dense crowds |

### A. Camera Motion Compensation (CMC) in BoT-SORT
If the camera moves (e.g. handheld or mounted on a drone), the pixel coordinates of stationary objects shift. BoT-SORT corrects Kalman state coordinates using global image registration (affine/homography matching via keypoints):

$$\mathbf{x}'_t = \mathbf{T}_{cam} \mathbf{x}_t$$

Where $\mathbf{T}_{cam}$ is the computed camera motion transformation matrix.

### B. ByteTrack Association Logic
Rather than discarding low-confidence detections (which might be partially occluded objects), ByteTrack matches them in a secondary association step. First, high-confidence detections are matched. Then, unmatched trackers are compared to the remaining low-confidence detections to recover occluded targets.

---

## 3. Comprehensive Parameter Details for `model.track()`

| Parameter | Type | Default Value | Range / Choices | Deep Technical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `source` | `str/int/array` | *Required* | Video path or stream | The video stream or frame buffer to run tracking on. |
| `tracker` | `str` | `'botsort.yaml'` | `'botsort.yaml'`, `'bytetrack.yaml'` | The tracker configuration file to use. |
| `persist` | `bool` | `False` | `True`, `False` | Retains tracker states across consecutive frames. Must be set to `True` when calling tracking iteratively in a loop. |
| `conf` | `float` | `0.25` | $[0.0, 1.0]$ | Confidence threshold for the detection step. |
| `iou` | `float` | `0.7` | $[0.0, 1.0)$ | IoU threshold for NMS. |

---

## 4. Python vs. CLI Code Comparison

### Python API Usage
```python
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("traffic.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Run tracking. persist=True maintains identity memory across frames
    results = model.track(source=frame, tracker="bytetrack.yaml", persist=True)
    
    # Retrieve active IDs
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().numpy()
        print(f"Active Track IDs in frame: {track_ids}")
```

### CLI Usage
```bash
# Run tracking on video and save output directly to disk
yolo detect track model=yolo11n.pt source="traffic.mp4" tracker="bytetrack.yaml" save=True
```

---

## 💡 Professor Tips

### 1. Tracking Persistence Failure
If you run tracking in a loop but fail to specify `persist=True`, the tracker reinitializes on every frame. This resets all tracking IDs to 1, causing identity switches. Always check:
```python
results = model.track(source=frame, persist=True)  # CORRECT
```

### 2. Customizing Tracker Settings
To modify parameters like the number of frames to keep lost tracks (`track_buffer`) or detection thresholds, create a custom YAML file (copied from the default configuration files) and reference it in the tracking command:
```yaml
# custom_bytetrack.yaml
track_thresh: 0.5
track_buffer: 60  # Keep lost tracks for 60 frames before deletion
match_thresh: 0.8
```
Then execute:
```python
results = model.track(source=frame, tracker="custom_bytetrack.yaml", persist=True)
```

---
*Related Topics:*
*   [[EX60_Streaming_Inference|EX60: Streaming Inference]]
*   [[EX62_Inference_Visualization|EX62: Inference Visualization]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
