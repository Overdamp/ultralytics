# Exercise 1: Custom Inference Script

## 🎯 Task Goal
Write a simple Python script to load the trained YOLO weights (`runs/detect/train-3/weights/best.pt`) and run inference on one test image, printing out the class name and coordinates of the detected objects.

---

## 🛠️ Solution Implementation
Below is the python solution code:

```python
from ultralytics import YOLO

# 1. Load trained model
model = YOLO('runs/detect/train-3/weights/best.pt')

# 2. Run inference on a test image
results = model('datasets/coco8/overall-ptt-object-detection.v11i.yolov11/test/images/some_image.jpg')

# 3. Parse and print results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        coordinates = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        print(f"Class: {class_name} | Confidence: {confidence:.2f} | Coordinates: {coordinates}")
```

---
*Related Topics:*
*   [[Object_Detection_Tips]]
*   Back to [[learning_journal]]
