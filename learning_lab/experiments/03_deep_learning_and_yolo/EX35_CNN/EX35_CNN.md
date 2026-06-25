# 🧠 EX35: Convolutional Neural Networks (CNN)

Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed for processing grid-like data, such as 2D images. They utilize spatial weight sharing to extract features efficiently.

---

## 1. Why CNNs over MLPs for Images?
If we pass a $640 \times 640$ RGB image into a fully connected Multi-Layer Perceptron (MLP):
1.  **Flattening destroys space:** Flattening a $640\times640\times3$ image into a 1D vector of $1,228,800$ values destroys all local spatial relationships (which pixels are next to each other).
2.  **Parameter Explosion:** If the first hidden layer has 1,000 neurons, we would need:
    $$1,228,800 \times 1,000 = 1.22 \text{ Billion Weights}$$
    This leads to massive overfitting and memory limits.

CNNs solve this using **Local Receptive Fields** (connecting neurons to local pixel regions) and **Weight Sharing** (using the same filter across the entire image).

---

## 2. Mathematical Formulation

### A. The Convolution Operation
A filter (or kernel) is a small matrix of weights (e.g., $3 \times 3$) that slides across the input image. At each step, it performs element-wise multiplication and sums the results:

$$O(i,j) = (I * K)(i,j) = \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} I(i + m, j + n) \cdot K(m, n)$$

Where $I$ is the input image, $K$ is the kernel of size $F \times F$, and $O$ is the output feature map.

### B. Output Shape Formula
The dimensions of the output feature map (width/height) depend on the input size ($W$), kernel size ($F$), padding ($P$), and stride ($S$):

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - F + 2P}{S} \right\rfloor + 1$$

*   **Stride ($S$):** The step size when sliding the kernel. Stride $2$ downsamples the output size by half.
*   **Padding ($P$):** Zero-values added to the border. It prevents spatial dimensions from shrinking at every layer and preserves edge details.

---

## 💻 Python PyTorch Convolution Example

Here is how to run a convolution in PyTorch and verify the output dimensions using the formula:

```python
import torch
import torch.nn as nn

# Create a mock image tensor: Batch=1, Channels=3 (RGB), 640x640 resolution
x = torch.randn(1, 3, 640, 640)

# Define a Convolutional Layer:
# - Input channels = 3
# - Output channels (filters) = 16
# - Kernel size = 3x3
# - Stride = 2
# - Padding = 1
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)

# Run forward pass
output = conv(x)
print(f"Input Shape:  {x.shape}")       # Output: torch.Size([1, 3, 640, 640])
print(f"Output Shape: {output.shape}")  # Output: torch.Size([1, 16, 320, 320])

# Verification using formula:
# W_out = floor((640 - 3 + 2(1)) / 2) + 1
#       = floor(639 / 2) + 1
#       = 319 + 1 = 320
```

---
*Related Topics:*
*   [[EX34_Backpropagation\|EX34: Backpropagation]]
*   [[EX36_Feature_Maps\|EX36: Feature Maps]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
