# 🧠 EX36: Feature Maps in CNNs

A Feature Map (also called an activation map) is the output of a convolutional layer. It represents the spatial distribution of specific visual features (e.g., edges, corners, textures, or semantic objects) detected by the layer's kernels.

---

## 1. Feature Map Dimensions: $[C, H, W]$
A feature map is represented as a 3D tensor:
*   **$C$ (Channels):** The number of filters/kernels applied in the convolutional layer. Each channel corresponds to a unique visual feature (e.g., one channel might highlight horizontal edges, while another highlights metallic reflection).
*   **$H \times W$ (Height and Width):** The spatial resolution of the feature maps, indicating where the detected features are located in the image.

---

## 2. The Feature Map Hierarchy
As an image passes deeper into a CNN backbone (like YOLO's CSPDarknet feature extractor), the dimensions of the feature maps change:

```
[ Input Image ] ---> [ Shallow Layers ] -------------> [ Deep Layers ]
  (640x640x3)        - Large H & W, Small C            - Small H & W, Large C
                     - High spatial resolution         - Low spatial resolution
                     - Low semantic abstraction        - High semantic abstraction
                     - (Detects lines, circles)        - (Detects whole valves, pumps)
```

*   **Shallow Layers:** Retain high localization precision, making them ideal for predicting precise bounding box borders.
*   **Deep Layers:** Lose spatial details (due to stride downsampling), but acquire high semantic context, making them ideal for predicting class labels.

---

## 🔬 Computer Vision Case Study: Visualizing YOLO Activations
If you extract and plot the feature maps of a YOLO model trained on [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11):
1.  **Layer 1 (Conv):** Activations will light up along the sharp linear edges of pipelines and the outer borders of `analog-gauges`.
2.  **Layer 15 (SPPF/Neck):** Spatial details disappear. Instead, the activations look like heatmaps centered directly on target objects, such as a strong activation bubble located on a `control-valve` body.

---

## 💻 Python PyTorch Feature Map Extraction

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel, 8 output feature map channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv1(x)

# Create model and dummy image (Batch=1, Channels=1, 64x64)
model = SimpleConvNet()
img = torch.randn(1, 1, 64, 64)

# Extract feature maps
with torch.no_grad():
    feature_maps = model(img) # Shape: [1, 8, 64, 64]

print(f"Feature Map Shape: {feature_maps.shape}")

# To plot the 3rd feature map channel:
# map_to_plot = feature_maps[0, 2, :, :].numpy()
# plt.imshow(map_to_plot, cmap='gray')
# plt.savefig('feature_map_3.png')
```

---
*Related Topics:*
*   [[EX35_CNN\|EX35: Convolutional Neural Networks]]
*   [[EX37_Pooling\|EX37: Pooling Layers]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
