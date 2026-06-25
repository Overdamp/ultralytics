# 🧠 EX08: Random Forest Theory

In this topic, we examine **Random Forests**, a powerful ensemble learning method that improves generalization by combining multiple decision trees.

---

## 1. Core Concept
A single Decision Tree is prone to **Overfitting** (low bias, but high variance). Random Forest resolves this by building a large ensemble of independent decision trees. 

To ensure the trees are diverse, it uses two techniques:
1.  **Bootstrap Aggregating (Bagging):** Each tree is trained on a random sample of the dataset drawn with replacement.
2.  **Random Feature Selection:** At each node split, only a random subset of features is considered, preventing dominant features from dictating every tree.

At prediction time, the outputs of all trees are aggregated: **majority voting** for classification, or **mean average** for regression.

---

## 2. Mathematical Formulation

### A. Bagging (Bootstrap Aggregation)
Given a training set $D$ of size $m$, we generate $B$ bootstrap datasets $D_b$ by sampling $m$ points uniformly with replacement. We train a separate tree $f_b$ on each dataset.

The ensemble prediction is:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x) \quad \text{(for Regression)}$$

$$\hat{y} = \text{argmax}_{c} \sum_{b=1}^{B} \mathbb{I}(f_b(x) == c) \quad \text{(for Classification)}$$

Where $\mathbb{I}(\cdot)$ is the indicator function.

### B. Out-of-Bag (OOB) Error Estimation
Since bootstrap sampling draws examples with replacement, approximately $\approx 36.8\%$ of data points are left out of training for any given tree. These are called **Out-of-Bag (OOB)** samples. 

We can evaluate the model's accuracy on OOB samples directly during training, eliminating the need for a separate validation split.

---

## 🔬 Computer Vision Case Study: Classifying Detections Under Noise
Suppose we classify valve types (e.g., `lever-valve` vs `handwheel-valve`) using geometric and color descriptors. Under varying camera angles, lighting conditions, or partial occlusions, a single decision tree might make wrong predictions because it overfits to particular thresholds.

### Applying Random Forest:
By aggregating predictions from $100$ trees—each trained on different variations of box sizes, color values, and subsets of features—the Random Forest filters out the noise. The majority voting ensures that classification remains stable even if a few features are completely occluded or distorted.

---
*Related Topics:*
*   [[EX07_Decision_Tree\|EX07: Decision Tree]]
*   [[EX09_SVM\|EX09: Support Vector Machines]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
