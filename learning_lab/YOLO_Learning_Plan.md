# 🗺️ YOLO Learning Plan & Roadmap

Welcome to your **Ultralytics YOLO** learning plan! This note is designed to help you track your progress, build custom computer vision models, and organize your studies. It connects seamlessly to other notes in your Obsidian Vault.

---

## 📅 Learning Schedule

| Duration | Learning Topic | Goal / Hands-on Task | References & Vault Links | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Day 1** | **YOLO Basics & Setup** | Learn CLI and Python API basics using local tutorial examples. | [[examples/tutorial.ipynb]]<br>[[Journal/learning_journal\|Daily Journal]] | 🟢 Completed |
| **Day 2-3** | **Dataset Preparation** | Understand YOLO annotation formats, labeling structures, and folder organization. | [[notebooks/YOLO_Training_Manual.ipynb\|Training Manual]] | 🟡 In Progress |
| **Day 4-5** | **Custom Model Training** | Fine-tune a custom YOLO model (26 classes) on your custom PTT dataset. | [[notebooks/YOLO_Training_Manual.ipynb\|Training Manual]] | ⚪ Not Started |
| **Day 6-7** | **Evaluation & Metrics** | Analyze train outputs in the `runs/` folder (mAP, Precision, Recall, Confusion Matrix). | [[notebooks/YOLO_Metrics_Explained.ipynb\|Metrics Guide]]<br>[[Concepts/Object_Detection_Tips\|Detection Tips]] | ⚪ Not Started |
| **Day 8-9** | **Hyperparameter Tuning** | Adjust learning rate, optimizers, and augmentations to combat overfitting. | [[notebooks/YOLO_Hyperparameters_Explained.ipynb\|Hyperparameter Guide]]<br>[[Concepts/YOLO_Loss_Functions\|Loss Functions]] | ⚪ Not Started |
| **Day 10** | **Export & Deployment** | Export weights to high-speed formats like ONNX or TensorRT for optimized runtime. | [[Exercises/ex01_inference\|Ex 1: Custom Inference]] | ⚪ Not Started |

---

## 🎯 Progress Checklist

- [x] **Setup & Verification:** Configured workspace customization for AI agents and organized directories.
- [x] **YOLO Basics:** Studied basic API workflows and modes in `tutorial.ipynb`.
- [ ] **Dataset Anatomy:** Inspected annotation structure and `data.yaml` inside `overall-ptt-object-detection.v11i.yolov11`.
- [ ] **Baseline Training:** Run first baseline training experiment and log results inside [[Journal/learning_journal\|Daily Journal]].
- [ ] **Metrics Insight:** Confidently explain difference between mAP@0.5 and mAP@0.5:0.95.
- [ ] **Custom Inference:** Successfully write custom Python script to run inference on test frames.

---

## 🧠 Vault Index
*   **Journal Logs:** [[Journal/learning_journal\|Daily log of experiments, bugs, and outcomes]]
*   **Core Concepts:**
    *   [[Concepts/YOLO_Loss_Functions\|YOLO Loss Functions (CIoU, DFL, BCE Loss Functions Explained)]]
    *   [[Concepts/Object_Detection_Tips\|Tips on Object Detection and Overfitting Prevention]]
*   **Exercises:**
    *   [[Exercises/ex01_inference\|Exercise 1: Basic Custom Inference Script]]

---
*Tip: Open the Graph View in Obsidian to visually explore how all these concepts and logs link together!*
