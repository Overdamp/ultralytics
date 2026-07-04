# Ultralytics Learning Journal & Progress Tracker

Use this journal to keep track of your learning milestones, model training experiments, and optimization results.

---

## 📅 Learning Log

| Date | Topics Studied / Tasks Done | Key Takeaways & Notes | Next Steps |
| :--- | :--- | :--- | :--- |
| 2026-06-25 | Setup environment & studied [tutorial.ipynb](file:///home/luke/ai_training/ultralytics/examples/tutorial.ipynb) | Setup learning workspace; learned about YOLO modes, Python API, and vision tasks (Detection, Segmentation, etc.). | Explore custom dataset training (`YOLO_Training_Manual.ipynb`). |
| 2026-07-03 | Implemented [[EX43_NMS]] content | Populated skeleton, solution, and explanation notebook for Non-Maximum Suppression (NMS). Analyzed IoU overlap suppression and Soft-NMS theory. | Move to [[EX44_Object_Detection_Pipeline\|EX44: Object Detection Pipeline]]. |
| 2026-07-03 | Implemented [[EX44_Object_Detection_Pipeline]] content | Populated skeleton, solution, and explanation notebook for Object Detection Pipeline Architecture. Modeled Backbone multi-scale maps, Neck feature fusion (FPN/PANet), and Decoupled Head class/box branches. | Move to [[EX45_Anchor_Box\|EX45: Anchor Boxes vs. Anchor-Free]]. |
| 2026-07-03 | Implemented [[EX45_Anchor_Box]] content | Populated skeleton, solution, and explanation notebook for Anchor-Based vs. Anchor-Free object detection. Modeled decoding formulas for both methods, analyzing how bounding box predictions are resolved from raw network values. | Move to [[EX46_YOLO_Grid\|EX46: YOLO Grid System]]. |
| 2026-07-03 | Implemented [[EX46_YOLO_Grid]] content | Populated skeleton, solution, and explanation notebook for YOLO Grid System. Modeled multiscale grid assignment logic (Strides 8, 16, and 32) and responsible grid cell indices mapping based on bounding box center and area. | Move to [[EX47_Confidence_Score\|EX47: Confidence Score]]. |
| 2026-07-03 | Implemented [[EX47_Confidence_Score]] content | Populated skeleton, solution, and explanation notebook for Confidence Score in Object Detection. Analyzed classic YOLO P(Class|Object)*P(Object)*IoU score and anchor-free task-aligned score s^alpha * IoU^beta. Modeled Precision-Recall trade-off shifts under different confidence thresholds. | Move to [[EX48_mAP\|EX48: Mean Average Precision (mAP)]]. |
| 2026-07-03 | Implemented [[EX48_mAP]] content | Populated skeleton, solution, and explanation notebook for Mean Average Precision (mAP). Designed single-class AP matching using IoU thresholds, calculated cumulative Precision & Recall, and implemented both 11-point and COCO-style all-point interpolation. Evaluated multiclass mock predictions for mAP@0.5 and mAP@0.5:0.95. | Move to [[EX49_YOLO_Result_Analysis\|EX49: YOLO Result Analysis]]. |
| 2026-07-03 | Implemented [[EX49_YOLO_Result_Analysis]] content | Populated skeleton, solution, and explanation notebook for YOLO Result Analysis. Designed pandas training log parser, best checkpoint identifier, and automated overfitting diagnostics using numpy polyfit slopes. Plotted train/val loss components (Box, Cls, DFL) and mAP curves. | Move to [[EX50_Hyperparameter_Tuning\|EX50: Hyperparameter Tuning]]. |
| 2026-07-03 | Implemented [[EX50_Hyperparameter_Tuning]] content | Populated skeleton, solution, and explanation notebook for YOLO Hyperparameter Tuning. Implemented Cosine Annealing learning rate decay scheduler and Genetic Algorithm mutation logic with parameter boundaries clipping. | Complete! Refer to [[YOLO_Learning_Plan]]. |
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
