# 🧠 EX07: Decision Tree Theory

In this topic, we examine **Decision Trees**, a versatile non-linear algorithm used for both classification and regression.

---

## 1. Core Concept
A Decision Tree acts like a flowchart. Starting at the root node, it splits the dataset into subsets based on feature values that maximize the purity of the resulting branches. This recursive splitting continues until a stopping criterion (e.g., maximum depth) is met at the leaf nodes.

---

## 2. Mathematical Formulation

To determine the best feature threshold to split on, we measure the "impurity" or disorder of the labels in a node.

### A. Entropy (Disorder)
Entropy measures the impurity of a dataset $S$. It is calculated as:

$$H(S) = -\sum_{i=1}^{c} p_i \log_2 p_i$$

Where:
*   $c$ is the number of classes.
*   $p_i$ is the probability/proportion of examples belonging to class $i$.
*   If a node is perfectly pure (all class 0), $H(S) = 0$. If it is split 50/50, $H(S) = 1$.

### B. Information Gain
We choose the split that decreases entropy the most. This reduction is called **Information Gain (IG)**:

$$IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where $A$ is the feature, and $S_v$ represents the subset of $S$ where feature $A$ takes value $v$.

### C. Gini Impurity (Alternative)
Often preferred over Entropy because it does not require computing logarithms (which speeds up training):

$$\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2$$

---

## 🔬 Computer Vision Case Study: Bounding Box Routing
Suppose you have trained your YOLO model on [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11) and you want to write a simple sorting script to route objects on a conveyor belt:

```
                  [ Root Node: Area > 15000px? ]
                             /         \
                           Yes          No
                           /              \
        [ Split 2: Aspect Ratio > 1.2? ]   [ Leaf: Small Fitting ]
                  /            \
                Yes             No
                /                 \
       [ Leaf: Control Valve ]  [ Leaf: Flange ]
```

A decision tree takes YOLO bounding box dimensions (`area`, `aspect_ratio`) and routes them automatically to the correct category.

---
*Related Topics:*
*   [[EX06_KNN\|EX06: K-Nearest Neighbors]]
*   [[EX08_Random_Forest\|EX08: Random Forest]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
