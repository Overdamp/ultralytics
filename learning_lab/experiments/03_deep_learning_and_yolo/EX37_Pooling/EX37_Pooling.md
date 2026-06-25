# 🧠 EX37: Pooling Layers in CNNs

Pooling layers downsample feature maps along their spatial dimensions (width and height), reducing the computational footprint of the network while building translation invariance.

---

## 1. Types of Pooling

### A. Max Pooling (Most Common)
Max pooling slides a window across the input feature map and extracts the **maximum value** in that window.
*   **Formula:**
    $$O_{i,j} = \max_{m,n \in W} I_{i \cdot s + m, \ j \cdot s + n}$$
    Where $s$ is the stride, $W$ is the window dimensions, and $I$ is the input grid.
*   **Visual Grid Example ($2 \times 2$ Max Pool, Stride 2):**
    ```text
    Input (4x4):                Output (2x2):
    [ 1  3 | 2  9 ]             [ 8 | 9 ]
    [ 8  4 | 1  5 ]  =======>   [ 7 | 6 ]
    ---------------
    [ 2  7 | 0  3 ]
    [ 6  5 | 2  6 ]
    ```

### B. Average Pooling
Average pooling calculates the **arithmetic mean** of the values inside the window. It is often used at the final layer of classification models (Global Average Pooling).
*   **Formula:**
    $$O_{i,j} = \frac{1}{|W|} \sum_{m,n \in W} I_{i \cdot s + m, \ j \cdot s + n}$$

---

## 2. Key Properties of Pooling

1.  **Translation Invariance:** Because max pooling selects the highest local activation, if an object shifts by a few pixels in the input image, the max-pooled features remain identical. This makes the CNN robust to slight position changes of objects.
2.  **No Parameters:** Unlike convolutional layers, pooling layers do not have weights or biases to learn. They are purely mathematical downsamplers.
3.  **Dimensionality Reduction:** Reduces feature map sizes, decreasing memory consumption during backpropagation.

---

## 🔬 Computer Vision Connection: SPPF in YOLO
Modern YOLO models place an **SPPF (Spatial Pyramid Pooling - Fast)** block at the end of the backbone. 

Instead of choosing a single pooling size, SPPF runs three successive $5 \times 5$ Max Pooling operations in series. By concatenating the original feature map with the outputs of these successive pooling stages, YOLO extracts multi-scale context (receptive fields of $5\times5$, $9\times9$, and $13\times13$) without altering the spatial resolution of the feature tensor.

---

## 💻 Python PyTorch Pooling Example

```python
import torch
import torch.nn as nn

# Mock feature map: Batch size 1, 3 channels, 4x4 resolution
x = torch.tensor([[[
    [1.0, 3.0, 2.0, 9.0],
    [8.0, 4.0, 1.0, 5.0],
    [2.0, 7.0, 0.0, 3.0],
    [6.0, 5.0, 2.0, 6.0]
]]])

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

print("Max Pool Output:\n", max_pool(x))
print("Avg Pool Output:\n", avg_pool(x))
```

---
*Related Topics:*
*   [[EX36_Feature_Maps\|EX36: Feature Maps]]
*   [[EX38_Transfer_Learning\|EX38: Transfer Learning]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
