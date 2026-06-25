# 🧠 EX48: Mean Average Precision (mAP)

Mean Average Precision (mAP) is the standard benchmark metric used to evaluate object detection models. It measures the average accuracy of predicted bounding boxes across all classes.

---

## 1. Why Not Use Accuracy?
In object detection, a model must predict both the **class label** and the **spatial bounding box coordinates**. Standard classification accuracy fails because:
*   We need to determine if a box is placed correctly (overlap).
*   Images contain a high proportion of background pixels (massive class imbalance).

---

## 2. Step-by-Step mAP Calculation

To calculate mAP, we follow these steps:

### Step 1: Define Bounding Box Matches (IoU)
A predicted bounding box is considered a **True Positive (TP)** only if:
1.  The predicted class label matches the ground truth label.
2.  The overlap (Intersection over Union) with the ground truth box is greater than or equal to a threshold (e.g., $\text{IoU} \ge 0.5$).
Otherwise, it is labeled a **False Positive (FP)**.

### Step 2: Plot the Precision-Recall (PR) Curve
For a single class:
1.  Sort all model predictions by their confidence score in descending order.
2.  Calculate Precision and Recall progressively down the list.
3.  Plot these values to create the **Precision-Recall Curve**.

### Step 3: Calculate Average Precision (AP)
AP represents the area under the PR curve. Under the COCO standard, it is computed using the **all-point interpolation** method:

$$\text{AP} = \sum_{n} (R_{n+1} - R_n) P_{\text{interp}}(R_{n+1})$$

Where $P_{\text{interp}}(R) = \max_{\tilde{R} \ge R} P(\tilde{R})$ is the maximum precision for any recall value greater than $R$.

### Step 4: Calculate mAP
mAP is simply the mean of the AP values calculated across all $C$ classes:

$$\text{mAP} = \frac{1}{C} \sum_{i=1}^{C} \text{AP}_i$$

---

## 3. YOLO mAP Variations

In YOLO logs (e.g., `results.csv`), you will see two variations:
*   **mAP@0.5 (or mAP50):** The average AP across all classes calculated at a single IoU threshold of **0.5**. This measures how well the model finds and identifies objects.
*   **mAP@0.5:0.95 (or mAP50-95):** The AP averaged over 10 different IoU thresholds (0.50, 0.55, 0.60, ..., 0.95 in steps of 0.05). This is a much stricter metric that measures **localization precision** (how tightly the predicted boxes fit the objects).

---

## 🔬 Computer Vision Case Study: PTT Dataset Evaluation
Suppose you evaluate your model on the 26-class [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11) dataset.

If the model detects:
*   `control-valve`: $\text{AP} = 0.90$ (highly distinct shape, easy to detect)
*   `small-valve`: $\text{AP} = 0.40$ (often confused with fittings/pipes)
*   `flange`: $\text{AP} = 0.80$

The mAP over these three classes is:
$$\text{mAP} = \frac{0.90 + 0.40 + 0.80}{3} = 0.70 \quad (70\%)$$

A low class-specific AP tells you exactly which classes need more training data or hyperparameter tuning.

---
*Related Topics:*
*   [[EX47_Confidence_Score\|EX47: Confidence Score]]
*   [[EX49_YOLO_Result_Analysis\|EX49: YOLO Result Analysis]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
