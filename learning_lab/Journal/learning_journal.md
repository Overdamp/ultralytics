# Ultralytics Learning Journal & Progress Tracker

Use this journal to keep track of your learning milestones, model training experiments, and optimization results.

---

## 📅 Learning Log

| Date | Topics Studied / Tasks Done | Key Takeaways & Notes | Next Steps |
| :--- | :--- | :--- | :--- |
| 2026-06-25 | Setup environment & studied [tutorial.ipynb](file:///home/luke/ai_training/ultralytics/examples/tutorial.ipynb) | Setup learning workspace; learned about YOLO modes, Python API, and vision tasks (Detection, Segmentation, etc.). | Explore custom dataset training (`YOLO_Training_Manual.ipynb`). |

---

## 📚 Study Notes: `tutorial.ipynb` Summary

The tutorial notebook introduces the fundamentals of the **Ultralytics YOLO** (specifically **YOLO26**) framework.

### 1. Installation & Environment Check
*   **Install:** `pip install ultralytics`
*   **Verification:** `import ultralytics; ultralytics.checks()` checks software, CUDA/GPU hardware, and paths.

### 2. Core Operational Modes
YOLO can be used via the Command Line Interface (CLI) or the Python API for four main actions:
1.  **Predict:** Run object detection inference on images, videos, directories, or URLs.
    *   *CLI:* `yolo predict model=yolo26n.pt source=...`
2.  **Val:** Evaluate model precision, recall, and mean Average Precision (mAP) on validation datasets.
    *   *CLI:* `yolo val model=yolo26n.pt data=coco8.yaml`
3.  **Train:** Fine-tune pre-trained models or train new models from scratch on custom datasets.
    *   *CLI:* `yolo train model=yolo26n.pt data=coco8.yaml epochs=3 imgsz=640`
4.  **Export:** Convert PyTorch weights (`.pt`) into optimized formats like ONNX, TensorRT, or OpenVINO for faster deployment.
    *   *CLI:* `yolo export model=yolo26n.pt format=onnx`

### 3. Python API Integration
The Python API provides cleaner integration than running raw shell commands:
```python
from ultralytics import YOLO

# 1. Load Model (Pre-trained)
model = YOLO('yolo26n.pt') 

# 2. Train on dataset
results = model.train(data='coco8.yaml', epochs=3)

# 3. Validate performance
metrics = model.val()

# 4. Run inference
predictions = model('https://ultralytics.com/images/bus.jpg')

# 5. Export for deployment
model.export(format='onnx')
```

### 4. Supported Vision Tasks
Different suffixes on model names determine the specific vision task being addressed:
*   **Object Detection:** Draws bounding boxes around objects. (e.g., `yolo26n.pt`)
*   **Instance Segmentation (`-seg`):** Classifies pixels belonging to objects. (e.g., `yolo26n-seg.pt`)
*   **Image Classification (`-cls`):** Classifies the whole image. (e.g., `yolo26n-cls.pt`)
*   **Pose Estimation (`-pose`):** Detects keypoints/joints on human bodies. (e.g., `yolo26n-pose.pt`)
*   **Oriented Bounding Boxes (`-obb`):** Predicts rotated bounding boxes (ideal for aerial/satellite imaging). (e.g., `yolo26n-obb.pt`)

---

## 💡 Notes & Questions
*   *Write down any questions about loss functions, custom datasets, or server deployment here to ask the agents later!*
