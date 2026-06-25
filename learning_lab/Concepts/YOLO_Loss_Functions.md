# YOLO Loss Functions

Loss functions evaluate how well the model is predicting compared to ground-truth boxes. YOLOv8 and YOLO11 use three main loss terms:

## 1. Complete IoU (CIoU) Loss
*   **Purpose:** Measures the difference between the predicted bounding box and ground-truth bounding box.
*   **Key factors evaluated:**
    1.  **Overlap:** Intersection over Union (IoU).
    2.  **Center Distance:** Normalized distance between bounding box centers.
    3.  **Aspect Ratio:** Scale difference between widths and heights.

## 2. Distribution Focal Loss (DFL)
*   **Purpose:** Helps the model locate precise box boundaries when they are blurry or occluded.
*   **Mechanism:** Rather than treating coordinates as single regression targets, DFL models the boundary coordinates as a probability distribution.

## 3. Binary Cross-Entropy (BCE) Loss
*   **Purpose:** Used for class probabilities. Evaluates whether the predicted object category (e.g. `control-valve` or `meter`) matches the true label.

---
*Related Topics:*
*   [[Object_Detection_Tips]]
*   Back to [[learning_journal]]
