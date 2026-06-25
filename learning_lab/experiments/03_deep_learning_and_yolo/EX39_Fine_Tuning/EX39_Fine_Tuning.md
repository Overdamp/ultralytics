# 🧠 EX39: Fine-Tuning Models

Fine-Tuning is the process of taking a model that has already been trained on a general dataset (e.g., COCO) and training it further on a smaller, custom dataset (e.g., your PTT dataset) using low learning rates, often while freezing early layers.

---

## 1. Fine-Tuning vs. Transfer Learning
*   **Transfer Learning** is the broad design pattern of applying knowledge gained from one task to a different but related task.
*   **Fine-Tuning** is a specific, hands-on implementation strategy of transfer learning, where we keep the weights of the pre-trained model and update them slowly during custom training.

---

## 2. The Fine-Tuning Workflow

```
[ Load Pre-trained Weights ] ---> [ Swap Classification Head ] ---> [ Freeze Backbone Layers ] ---> [ Train with Low Learning Rate ]
```

### Step 1: Replace the Classifier Head
Since the pre-trained model was trained on a different number of classes (e.g., COCO has 80 classes, but your PTT dataset has 26), the final output layer of the network is removed and replaced with a new, randomly initialized head matching your class count.

### Step 2: Freeze Feature Extraction Layers (Optional)
The early layers of the network (the backbone) detect basic geometric features like lines, corners, circles, and gradients. These features are universal across most images. By **freezing** these layers (setting `requires_grad = False`), we prevent backpropagation from changing them, saving computation and training time.

### Step 3: Low Learning Rate Optimization
The remaining unfrozen layers are trained using a very small learning rate (typically $\approx 10^{-4}$ or $10^{-5}$). This prevents the model from overwriting its pre-trained knowledge too aggressively, which is known as **Catastrophic Forgetting**.

---

## 🔬 Computer Vision Case Study: Freezing YOLO Backbone
When fine-tuning YOLO on the 26-class [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11) dataset, we can freeze the first $N$ layers of the backbone:

```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolo26n.pt')

# Fine-tune the model, freezing the first 10 layers (the backbone)
# This protects the base feature extraction weights and only trains the neck/head.
model.train(
    data='datasets/coco8/overall-ptt-object-detection.v11i.yolov11/data.yaml',
    epochs=30,
    lr0=0.001,      # Lower learning rate than scratch default (0.01)
    freeze=10       # Freeze first 10 layers
)
```

---
*Related Topics:*
*   [[EX38_Transfer_Learning\|EX38: Transfer Learning]]
*   [[EX40_Embeddings\|EX40: Feature Embeddings]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
