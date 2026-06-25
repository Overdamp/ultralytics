# 🧠 EX09: Support Vector Machines (SVM) Theory

In this topic, we examine **Support Vector Machines (SVM)**, a robust classification method that finds the optimal separating hyperplane in high-dimensional feature spaces.

---

## 1. Core Concept
The goal of an SVM is to find a decision boundary (hyperplane) that separates two classes while maximizing the **margin** (the distance between the boundary and the closest data points of either class, known as **support vectors**).

If the data is not linearly separable in its original space, SVM maps the features into a higher-dimensional space where they become separable. This is done efficiently using the **Kernel Trick**.

---

## 2. Mathematical Formulation

### A. The Optimization Problem (Hard Margin)
A hyperplane is defined as $\mathbf{w}^T \mathbf{x} + b = 0$. For linearly separable training data:

$$y^{(i)} (\mathbf{w}^T \mathbf{x}^{(i)} + b) \ge 1 \quad \forall i$$

To maximize the margin (which is $\frac{2}{\|\mathbf{w}\|}$), we minimize the magnitude of the weights:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to } y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \ge 1$$

### B. Soft Margin (Dealing with Noise)
If there is overlap, we introduce slack variables $\xi_i$ to allow minor violations of the margin:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \xi_i$$

Where $C$ is a regularization hyperparameter:
*   A large $C$ heavily penalizes misclassifications, leading to a narrower margin (higher variance, lower bias).
*   A small $C$ allows more classification errors, yielding a wider margin (lower variance, higher bias).

### C. The Kernel Trick (Non-linear boundaries)
Instead of manually mapping feature vectors to higher dimensions, we use kernel functions $K(\mathbf{x}, \mathbf{z})$ that compute the dot product in the higher-dimensional space directly:
*   **Radial Basis Function (RBF) Kernel:**
    $$K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$$

---

## 🔬 Computer Vision Case Study: Valve-Stem vs. Polished-Rod
Suppose you want to classify two visual features that are highly similar in structure: a `polished-rod` vs. a `manual-valve-stem` from [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11). 

Since both are cylindrical metal stems, simple linear features might overlap. An SVM with an **RBF Kernel** takes high-dimensional edge histograms or CNN features of the bounding box and draws a non-linear decision boundary to separate them with high confidence.

---
*Related Topics:*
*   [[EX08_Random_Forest\|EX08: Random Forest]]
*   [[EX10_Naive_Bayes\|EX10: Naive Bayes]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
