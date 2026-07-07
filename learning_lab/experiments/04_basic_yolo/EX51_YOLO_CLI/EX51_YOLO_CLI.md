# 🧠 EX51: YOLO Command Line Interface (CLI)

The Command Line Interface (CLI) is one of the most powerful and convenient ways to interact with Ultralytics YOLO. It allows you to train, validate, predict, export, track, and benchmark models directly from the terminal without writing a single line of Python code. However, as advanced computer vision practitioners, we must understand the engineering mechanics under the hood—specifically, how the OS manages processes, CUDA memory allocation overhead, and how the CLI compares with the Python API.

---

## 1. CLI Syntax Structure and Command Execution

The Ultralytics YOLO CLI follows a simple, intuitive syntax:

```bash
yolo TASK MODE ARGS
```

This structure consists of three main components:
1. **`yolo`**: The global entrypoint command mapped to the `ultralytics.cfg.entrypoint` function.
2. **`TASK`**: Specifies the computer vision task (optional in many cases, as YOLO can infer it from the model name suffix). Choices: `detect`, `segment`, `classify`, `pose`, `obb`.
3. **`MODE`**: Specifies the operational phase. Choices: `train`, `val`, `predict`, `export`, `track`, `benchmark`.
4. **`ARGS`**: Custom arguments in the form of `key=value` pairs to override default configurations (e.g., `epochs=50`, `imgsz=640`, `device=0`).

---

## 2. Core Components & Computer Vision Tasks

From a first-principles perspective, each `TASK` represents a distinct mathematical output head of the neural network:

### A. Task Types (`TASK`)
*   **Object Detection (`detect`)**: Solves the bounding box regression problem. The network outputs a tensor of shape $[B, A, 4 + C]$, where $B$ is batch size, $A$ is the number of anchor points or bounding boxes, $4$ represents the coordinates $(x_c, y_c, w, h)$, and $C$ represents class scores.
*   **Instance Segmentation (`segment`)**: Predicts class labels, bounding boxes, and pixel-level masks. It outputs bounding boxes alongside a set of prototype masks and coefficients to reconstruct precise shapes:
    $$\mathbf{M} = \sigma\left( \sum_{i=1}^k c_i \mathbf{P}_i \right)$$
*   **Image Classification (`classify`)**: Projects the features extracted from the backbone into a single classification probability vector using Softmax:
    $$P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{z}_c}}{\sum_{j=1}^K e^{\mathbf{z}_j}}$$
*   **Pose Estimation (`pose`)**: Regresses the 2D coordinates $(x_k, y_k)$ and visibility probability $v_k$ of $K$ predefined keypoints:
    $$\text{Keypoints} = \{(x_k, y_k, v_k)\}_{k=1}^K$$
*   **Oriented Object Detection (`obb`)**: Regresses rotated bounding boxes containing an angle parameter $\theta$ to capture objects at arbitrary orientations (e.g., aerial photography):
    $$\mathbf{B}_{obb} = [x_c, y_c, w, h, \theta]$$

### B. Mode Types (`MODE`)
*   **`train`**: Fits the model weights to the target dataset using backward propagation and optimizer steps.
*   **`val`**: Calculates evaluation metrics (Precision, Recall, mAP) on a labeled validation set.
*   **`predict`**: Performs forward-pass inference on unlabeled data and outputs predictions.
*   **`export`**: Translates the PyTorch computation graph (`.pt`) into target formats (ONNX, TensorRT, OpenVINO) for faster execution.
*   **`track`**: Tracks objects over video frames, maintaining persistent IDs via tracking algorithms (BoT-SORT, ByteTrack).
*   **`benchmark`**: Benchmarks speed (ms per image) and accuracy across different export formats.

---

## 3. CLI Options and Hyperparameters Reference

The table below lists the most common arguments used in YOLO CLI commands:

| Parameter | Type | Default | Choices / Format | Deep Technical Description |
| :--- | :--- | :--- | :--- | :--- |
| `model` | `str` | `None` | Path to `.pt` or `.yaml` | The path to the model weights or architecture definition file. |
| `data` | `str` | `None` | Path to `.yaml` | Path to the dataset configuration file (contains class names and train/val/test splits). |
| `epochs` | `int` | `100` | $[1, \infty)$ | Number of complete passes over the training dataset. |
| `batch` | `int` | `16` | $[-1, \infty)$ | Number of samples processed in a single forward/backward pass. `-1` triggers auto-batching. |
| `imgsz` | `int/tuple` | `640` | Multiples of 32 | Resolution target of the input images (e.g., `imgsz=640` or `imgsz=480,640`). |
| `device` | `str/int` | `None` | `cpu`, `0`, `0,1`, `cuda` | Target hardware. Numeric values select CUDA devices. Lists activate Distributed Data Parallel (DDP). |
| `save` | `bool` | `True` | `True`, `False` | Save annotated visual results to the runs directory. |
| `show` | `bool` | `False` | `True`, `False` | Display results interactively in an OpenCV graphical window. |
| `half` | `bool` | `False` | `True`, `False` | Use half-precision floating-point (FP16) operations to reduce memory. |

---

## 4. Python API vs. CLI: Comparative Code Blocks

Every operations sequence can be completed using either the command line or Python code:

### Object Detection Training
*   **CLI**:
    ```bash
    yolo detect train model=yolo26n.pt data=coco8.yaml epochs=5 imgsz=640 device=0
    ```
*   **Python**:
    ```python
    from ultralytics import YOLO
    
    model = YOLO("yolo26n.pt")
    model.train(data="coco8.yaml", epochs=5, imgsz=640, device=0)
    ```

### Instance Segmentation Prediction
*   **CLI**:
    ```bash
    yolo segment predict model=yolo26n-seg.pt source="bus.jpg" save=True
    ```
*   **Python**:
    ```python
    from ultralytics import YOLO
    
    model = YOLO("yolo26n-seg.pt")
    results = model.predict(source="bus.jpg", save=True)
    ```

### Model Export to ONNX
*   **CLI**:
    ```bash
    yolo export model=yolo26n.pt format=onnx imgsz=640 half=True
    ```
*   **Python**:
    ```python
    from ultralytics import YOLO
    
    model = YOLO("yolo26n.pt")
    model.export(format="onnx", imgsz=640, half=True)
    ```

---

## 👨‍🏫 CLI vs. Python API: Architectural Deep-Dive

As a computer vision engineer, you must know what happens under the hood when you execute a command.

### 1. Process Life Cycle and Spawning Overhead
*   **CLI Execution (`yolo ...`)**: Spawns a standalone OS process. Every time the CLI is run, the Python interpreter starts, imports standard libraries (PyTorch, OpenCV, NumPy), initializes CUDA drivers, and allocations are made from scratch. This introduces a cold-start overhead of **$1.5\text{--}3.0\text{ seconds}$** before a single frame is processed.
*   **Python API Execution**: The imports and initialization occur once when you import the module and instantiate the `YOLO` object. Subsequent calls inside loops execute with **sub-millisecond overhead**, bypassing the process initialization code entirely.

### 2. VRAM and System Memory Retention
*   **CLI**: Once the process finishes, the operating system reclaims all system RAM and GPU VRAM immediately. This prevents memory leaks.
*   **Python API**: The model weights, gradients, and intermediate tensors remain resident in the PyTorch CUDA memory pool until explicitly garbage-collected or until the Python process terminates.

---

## 💡 Professor Tips

### 1. Auto-Inference of Task Suffixes
You do not need to explicitly specify the `task` argument if your model weight files follow standard naming. YOLO reads suffixes like `-seg`, `-pose`, and `-cls` to map the task:
```bash
# Inferred task: segment
yolo train model=yolo26n-seg.pt data=custom.yaml
```

### 2. Managing GPU Selection via Environment Variables
If your server has multiple GPUs and you want to ensure YOLO only accesses specific devices, set the `CUDA_VISIBLE_DEVICES` environment variable prior to executing the CLI:
```bash
CUDA_VISIBLE_DEVICES=1 yolo predict model=yolo26n.pt source=bus.jpg
```
This forces the PyTorch backend to treat GPU 1 as logical device `0`, preventing accidental resource collisions.

### 3. Preventing OOM on Headless CLI Runs
When deploying CLI commands inside remote shell runners or CI/CD pipelines, always verify that `show=False` (default) and `save=True`. If `show=True` is executed in a headless server environment, the CLI will throw an OpenCV GUI crash.

---

*Related Topics:*
*   [[EX52_Model_Configurations|EX52: Model Configurations]]
*   [[EX53_Training_Settings|EX53: Training Settings]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
