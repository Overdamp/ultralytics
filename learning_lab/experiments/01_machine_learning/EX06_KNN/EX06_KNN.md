# 🧠 EX06: K-Nearest Neighbors (KNN) Theory

In this topic, we examine **K-Nearest Neighbors (KNN)**, an intuitive, non-parametric, instance-based learning algorithm.

---

## 1. Core Concept
KNN is a **lazy learner**, meaning it does not perform any explicit training phase. Instead, it memorizes the training dataset. 

To classify a new query point, KNN:
1.  Calculates the distance between the query point and all points in the training set.
2.  Selects the **$K$** closest points (neighbors).
3.  Performs a **majority vote** (classification) or calculates the average (regression) to determine the output.

---

## 2. Mathematical Formulation

### A. Distance Metrics
The performance of KNN depends heavily on the distance metric. The most common choice is the **Euclidean Distance**:

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{d} (q_i - p_i)^2}$$

Where $\mathbf{p}$ and $\mathbf{q}$ are two data points in $d$-dimensional feature space.

Alternative metrics include **Manhattan Distance** (L1 norm):
$$d(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^{d} |q_i - p_i|$$

### B. The Choice of K
*   If $K$ is too small (e.g. $K=1$), the model is highly sensitive to noise, causing **Overfitting** (low bias, high variance).
*   If $K$ is too large, the model smooths out boundaries, causing **Underfitting** (high bias, low variance).
*   For binary classification, $K$ is typically chosen as an odd number to avoid tie-votes.

---

## 🔬 Computer Vision Case Study: Bounding Box Feature Classification
When a YOLO model detects an industrial component (e.g. a `lever-handle` or `handwheel-handle` in [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11)), we can extract a dense visual feature embedding vector (representation) using the neural network backbone.

### Applying KNN:
1.  Extract a 128-dimensional embedding vector $\mathbf{v}$ for a newly detected box.
2.  Compute the Euclidean distance between $\mathbf{v}$ and all reference embeddings in our database.
3.  Find the $K=5$ nearest neighbors. If 4 of them are labeled `lever-handle`, KNN classifies the new object as a `lever-handle`.

---
*Related Topics:*
*   [[EX05_Logistic_Regression\|EX05: Logistic Regression]]
*   [[EX07_Decision_Tree\|EX07: Decision Tree]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
