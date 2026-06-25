# 🧠 EX46: YOLO Grid System

The YOLO (You Only Look Once) grid system is the architectural concept that enables real-time single-stage object detection by dividing the image into cells that predict bounding boxes and classes simultaneously.

---

## 1. The Core Grid Concept
Classic YOLO divided the input image into a single grid (e.g., $S \times S = 7 \times 7$ or $13 \times 13$). 
*   **The Grid Cell Rule:** Whichever grid cell contains the center point of an object's ground truth bounding box is responsible for detecting that object.
*   This single-pass grid evaluation eliminates the need for expensive sliding windows or region proposal networks (like in Faster R-CNN).

---

## 2. Modern Multi-Scale Grid Resolution (FPN/PANet Strides)
In modern YOLO architectures (v8, v11), the model does not predict on a single grid. Instead, it extracts feature maps at three different resolution levels (scales) using strides of **8, 16, and 32**:

If the input image size is $640 \times 640$:

| Stride | Grid Dimensions | Cell Count | Target Object Size | Example from PTT Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **Stride 8** | $80 \times 80$ | 6,400 cells | **Small** (fine details) | `small-valve`, `pig-alert`, `digital-gauge` |
| **Stride 16** | $40 \times 40$ | 1,600 cells | **Medium** | `flange`, `lever-handle`, `control-valve` |
| **Stride 32** | $20 \times 20$ | 400 cells | **Large** (global features) | `wellhead`, `xmas-tree`, `pump` |

Each cell in these grids outputs:
1.  **Classification scores** for all 26 classes.
2.  **Bounding box regressors** (representing distances from the cell center to the left, right, top, and bottom edges of the box).

---

## 🔬 Computer Vision Case Study: Multi-Scale Grid Assignment
Imagine an input image containing a giant `wellhead` structure and a tiny `pig-alert` indicator:
1.  The center of the `wellhead` falls into a cell on the **$20 \times 20$ grid (Stride 32)**. The network uses this coarse grid layer to construct a large bounding box because the object spans a massive spatial area.
2.  The center of the `pig-alert` falls into a cell on the **$80 \times 80$ grid (Stride 8)**. The high resolution of this grid is essential for detecting the tiny, pixel-level features of the alert switch without losing them in downsampling.

---
*Related Topics:*
*   [[EX45_Anchor_Box\|EX45: Anchor Boxes vs. Anchor-Free]]
*   [[EX47_Confidence_Score\|EX47: Confidence Score]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
