# 🧠 EX10: Naive Bayes Theory

In this topic, we examine **Naive Bayes**, a probabilistic classifier based on Bayes' Theorem with a strong (naive) independence assumption between features.

---

## 1. Core Concept
Naive Bayes calculates the probability that a data point belongs to a class, given its features. It is called **"naive"** because it assumes that all features are completely independent of each other given the class, which simplifies the probability calculations enormously.

Despite this simplification, Naive Bayes performs surprisingly well on many complex classification tasks.

---

## 2. Mathematical Formulation

### A. Bayes' Theorem
To find the probability of class $C_k$ given an input feature vector $\mathbf{x} = (x_1, x_2, \dots, x_n)$:

$$P(C_k | \mathbf{x}) = \frac{P(\mathbf{x} | C_k) P(C_k)}{P(\mathbf{x})}$$

Where:
*   $P(C_k | \mathbf{x})$ is the **Posterior probability**.
*   $P(\mathbf{x} | C_k)$ is the **Likelihood** of features given the class.
*   $P(C_k)$ is the **Prior probability** of the class.
*   $P(\mathbf{x})$ is the **Evidence** (acting as a normalization constant).

### B. The Independence Assumption
Under the naive assumption that all features $x_i$ are independent given the class $C_k$, the joint probability model is:

$$P(\mathbf{x} | C_k) = P(x_1 | C_k) \cdot P(x_2 | C_k) \dots P(x_n | C_k) = \prod_{j=1}^{n} P(x_j | C_k)$$

This allows us to formulate the classifier as:

$$\hat{y} = \text{argmax}_{k} \left[ P(C_k) \prod_{j=1}^{n} P(x_j | C_k) \right]$$

---

## 🔬 Computer Vision Case Study: Bounding Box Filter
Suppose YOLO detects an object with $85\%$ model confidence, but you want to perform a double-check validation. You use historical prior frequencies to decide if a detection is a real object (`C=1`) vs background clutter/false positive (`C=0`) based on:
1.  `x1`: Bounding box height-to-width ratio.
2.  `x2`: Detection location (bounding box center coordinates).
3.  `x3`: Time of day (e.g. night camera feed has more noise).

### Calculating Posterior Probability:
Using Naive Bayes, you can compute:
$$P(\text{Real Object} | \mathbf{x}) \propto P(\text{Real Object}) \cdot P(x_1 | \text{Real}) \cdot P(x_2 | \text{Real}) \cdot P(x_3 | \text{Real})$$

If the posterior probability of it being background clutter is higher than it being a real object, you filter out the bounding box.

---
*Related Topics:*
*   [[EX09_SVM\|EX09: Support Vector Machines]]
*   [[EX11_Accuracy\|EX11: Accuracy]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
