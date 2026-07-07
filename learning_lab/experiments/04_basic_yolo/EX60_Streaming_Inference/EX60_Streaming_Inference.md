# 🧠 EX60: Streaming Inference

When processing long video files or continuous live network streams (such as RTSP, RTMP, or webcam feeds), running standard inference (`results = model(source)`) presents a critical vulnerability: **memory accumulation**.

By default, YOLO accumulates all prediction `Results` objects in system RAM, returning them as a single list after the source is fully processed. For a continuous 24/7 stream or a multi-gigabyte video, this will eventually exhaust system memory and trigger an **Out-Of-Memory (OOM)** crash. 

To solve this, YOLO provides a memory-efficient streaming mode via the `stream=True` parameter in `model.predict()`.

---

## 1. Mathematical and Computational Foundations of Streaming Inference

To understand why streaming is necessary, we must analyze the RAM consumption patterns and processing queues.

### A. Memory Complexity Analysis
Let $N$ be the number of frames in a video, and let $K_{bytes}$ represent the average memory footprint of the prediction metadata (bounding boxes, class probabilities, masks, and original frame arrays) for a single frame.

#### Standard Batch Inference (`stream=False`)
All output objects are retained in RAM simultaneously. The memory consumption at frame $t$ is:

$$M_{RAM}(t) = \sum_{i=1}^t K_{bytes} \approx t \cdot K_{bytes} \in \mathcal{O}(N)$$

For long streams ($N \rightarrow \infty$), memory usage grows linearly, leading to system crash.

#### Streaming Generator Inference (`stream=True`)
Only the current frame's result object is retained. Once the loop advances, the prior reference is dereferenced:

$$M_{RAM}(t) = K_{bytes} \in \mathcal{O}(1)$$

Memory usage remains constant and bounded, regardless of video length.

### B. Queueing Dynamics and Real-Time Latency
Let $R$ be the arrival frame rate of the stream (frames per second, FPS), and let $T_{proc}$ be the time (in seconds) the GPU/CPU takes to process a single frame.

*   If $T_{proc} \le \frac{1}{R}$, the model can process frames in real-time without latency accumulation.
*   If $T_{proc} > \frac{1}{R}$, a processing backlog forms. If frames are queued (`stream_buffer=True`), the latency $L(t)$ at time $t$ grows linearly:

$$L(t) = t \cdot \left( T_{proc} \cdot R - 1 \right)$$

To maintain real-time performance, older frames must be dropped (`stream_buffer=False`). The frame drop rate $F_{drop}$ is formulated as:

$$F_{drop} = 1 - \frac{1}{T_{proc} \cdot R}$$

---

## 2. Comprehensive Parameter Details for Streaming

| Parameter | Type | Default Value | Range / Choices | Deep Technical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `stream` | `bool` | `False` | `True`, `False` | Triggers generator mode. Returns a Python generator instead of loading all frame outputs into memory. |
| `vid_stride` | `int` | `1` | $[1, \infty)$ | Decimates frame processing rate. E.g., `vid_stride=3` only evaluates every 3rd frame. |
| `stream_buffer`| `bool` | `False` | `True`, `False` | When `True`, buffers incoming frames to prevent drops. When `False`, drops old frames to maintain low latency. |

---

## 3. Python vs. CLI Code Comparison

### Python API Usage
```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Process stream with memory safety
# stream=True returns a generator
generator = model.predict(source="traffic.mp4", stream=True)

for frame_idx, result in enumerate(generator):
    boxes = result.boxes
    print(f"Frame {frame_idx}: Detected {len(boxes)} objects.")
```

### CLI Usage
The CLI automatically runs streaming mode with optimized memory management.
```bash
# Process video using frame striding to speed up execution
yolo detect predict model=yolo11n.pt source="traffic.mp4" vid_stride=2 save=True
```

---

## 💡 Professor Tips

### 1. The Multi-Threaded Producer-Consumer Pattern
For production RTSP deployments, do not run frame decoding and model inference in the same thread. If model inference takes $30\text{ ms}$ and network packet retrieval delays frame decoding by $10\text{ ms}$, the GPU remains underutilized. Use a multi-threaded design:

```python
import queue
import threading
import cv2
from ultralytics import YOLO

class RTSPStreamer:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.q = queue.Queue(maxsize=2)  # Maintain short queue to prevent latency accumulation
        self.stopped = False
        
    def produce(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                break
            if not self.q.full():
                self.q.put(frame)
            else:
                try:
                    self.q.get_nowait()  # Drop oldest frame
                    self.q.put(frame)
                except queue.Empty:
                    pass
        cap.release()

    def consume(self, model_path):
        model = YOLO(model_path)
        while not self.stopped:
            if not self.q.empty():
                frame = self.q.get()
                results = model.predict(source=frame, verbose=False)
                # Downstream processing here
```

### 2. Guarding Against Memory Leaks
When running streaming generator inference (`stream=True`), avoid appending prediction items to lists outside the generator context. Storing reference logs (such as caching raw result tensors) defeats the $\mathcal{O}(1)$ memory guarantee, accumulating objects in memory and causing OOM crashes.

---
*Related Topics:*
*   [[EX56_Prediction_Sources|EX56: Prediction Sources]]
*   [[EX59_Multi_Task_Modes|EX59: YOLO Multi-Task Modes]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
