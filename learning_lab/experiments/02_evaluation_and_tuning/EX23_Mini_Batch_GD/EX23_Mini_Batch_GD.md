# 🧠 EX23: Mini-Batch Gradient Descent

Mini-Batch Gradient Descent is a variant of the gradient descent algorithm that splits the training dataset into small, manageable groups (mini-batches). The model weights are updated after evaluating each mini-batch, combining the advantages of both Batch GD and Stochastic GD.

---

## 1. Comparing Gradient Descent Variants

| Metric | Batch Gradient Descent | Stochastic Gradient Descent (SGD) | Mini-Batch Gradient Descent |
| :--- | :--- | :--- | :--- |
| **Batch Size** | Entire dataset ($m$) | Exactly 1 sample | Small group ($B$), e.g., 16, 32 |
| **Gradients** | Highly accurate, stable | Extremely noisy, erratic | Moderately stable, slightly noisy |
| **Speed** | Very slow for large data | Slow (cannot use vectorization) | Fast (uses GPU matrix acceleration) |
| **Memory** | High (often causes OOM) | Very low | Controlled (fits in VRAM) |
| **Generalization** | Can get stuck in local minima | Noise helps escape local minima | Balanced path, optimal generalization |

---

## 2. Mathematical Formulation

Let the mini-batch size be $B$ (where $1 < B < m$). 

For a given mini-batch containing training samples $\{ (x^{(1)}, y^{(1)}), \dots, (x^{(B)}, y^{(B)}) \}$, the loss function is calculated as the average loss over this subset:

$$J_{\text{mini-batch}}(w, b) = \frac{1}{2B} \sum_{i=1}^{B} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$

The parameters are updated using only the gradients computed from these $B$ samples:

$$w \leftarrow w - \alpha \frac{1}{B} \sum_{i=1}^{B} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)}$$
$$b \leftarrow b - \alpha \frac{1}{B} \sum_{i=1}^{B} \left( h_{w,b}(x^{(i)}) - y^{(i)} \right)$$

---

## 💻 Hardware Acceleration & Vectorization
Modern GPUs (NVIDIA Tesla, RT-cores) are optimized for parallel matrix multiplications. 
*   If we train with **SGD (Batch Size = 1)**, the GPU has to wait for sequential operations, leaving thousands of CUDA cores idle.
*   If we use **Mini-Batch GD (Batch Size = 16 or 32)**, we stack multiple images together into a 4D tensor of shape:
    $$\text{Shape} = [\text{Batch}, \text{Channels}, \text{Height}, \text{Width}]$$
    This allows the GPU to compute gradients for all 16 images in parallel, drastically reducing training times.

---

## 🔬 Computer Vision Connection: YOLO Training Batch
YOLO models use Mini-Batch GD during training. You control this in the command line or Python code using the `batch` argument (which defaults to `16`). 

```bash
yolo train model=yolo26n.pt data=data.yaml batch=16 imgsz=640
```
This partitions your custom PTT dataset images into groups of 16, maximizing GPU throughput while maintaining stable convergence.

---
*Related Topics:*
*   [[EX22_Stochastic_GD\|EX22: Stochastic Gradient Descent]]
*   [[EX24_Momentum\|EX24: Momentum Optimization]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
