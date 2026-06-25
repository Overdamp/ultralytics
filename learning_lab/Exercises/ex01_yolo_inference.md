# 🎯 Exercise 1: Custom Bounding Box Inference

## 📖 Task Description
Your goal is to write a standalone Python script to load a pre-trained YOLO weight file, run inference on a local image (e.g., `bus.jpg`), and print the detected bounding box details.

You will need to fill in the blanks in the accompanying Python script: [[ex01_yolo_inference.py\|ex01_yolo_inference.py]].

---

## 💡 Quick Theory Review
To run inference on an image in Python, we use the `YOLO` class:
1.  **Instantiation:** `model = YOLO("weight_path.pt")`
2.  **Inference:** `results = model("image_path.jpg")`
3.  **Result Extraction:** The output `results` is a list of `Results` objects. Each object contains a `.boxes` property containing details of predicted detections (class ID, coordinates, confidence).

For more tips on running inference and tuning detection, refer to:
*   [[Concepts/Object_Detection_Tips\|Object Detection Tips]]
*   [[Concepts/YOLO_Loss_Functions\|YOLO Loss Functions]]

---

## 🚀 How to Run
Once you fill in the blanks in the Python script, execute it from your terminal:
```bash
python learning_lab/Exercises/ex01_yolo_inference.py
```
Check [[Journal/learning_journal\|Daily Journal]] to log your execution results and notes!
