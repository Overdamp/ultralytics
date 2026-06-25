# 🧠 EX30: Epoch Analysis & Generalization Curves

An epoch represents one full pass of the training algorithm through the entire training dataset. Analyzing how loss metrics behave across epochs is essential to understand model learning rates and prevent overfitting.

---

## 1. Differentiating Epochs, Batches, and Iterations

To analyze training, we must understand how these three terms relate:
*   **Epoch:** One complete pass through the entire dataset.
*   **Batch Size:** The number of training examples processed in one forward/backward pass.
*   **Iteration (Step):** The execution of one gradient update.

$$\text{Iterations per Epoch} = \left\lceil \frac{\text{Total Training Samples}}{\text{Batch Size}} \right\rceil$$

### Example:
If you train on a custom dataset containing $800$ images with a `batch=16`:
$$\text{Iterations per Epoch} = \frac{800}{16} = 50 \text{ steps}$$
Training for $100$ epochs will run $5,000$ total gradient updates.

---

## 2. Generalization Curves & Overfitting Analysis
By plotting training and validation losses across epochs, we create **generalization curves** that diagnose the model's health:

```
Loss
  |
  |    \                 /  <-- Validation Loss starts climbing (Overfitting!)
  |     \               /
  |      \_____________/    <-- Optimal stopping point
  |       \           \
  |________\___________\_____
  0                           Epochs
```

1.  **Underfitting:** Both training loss and validation loss remain high and flat. The model has not learned the underlying patterns.
2.  **Overfitting:** Training loss continues to decline towards zero, but validation loss flattens and starts to increase. The model is memorizing the training images.
3.  **Optimal Stop (Early Stopping):** Training is halted when validation loss stops decreasing for a defined number of epochs (defined by the `patience` parameter, which defaults to `100` epochs in YOLO).

---

## 🔬 Computer Vision Case Study: YOLO Epoch Analysis
In your training run logs (e.g., `results.csv` and `results.png` inside [runs/detect/train-3](file:///home/luke/ai_training/ultralytics/runs/detect/train-3)):
*   Look at `val/box_loss` and `val/cls_loss`.
*   In `train-3`, the training completed 50 epochs.
*   Between epochs 40 and 50, the validation box loss remained steady at $\approx 1.38$, while training box loss fell from $1.49$ to $1.41$. This indicates that 50 epochs was the optimal length; training longer would risk overfitting.

---

## 💻 Python Generalization Curve Plotter

```python
import matplotlib.pyplot as plt

# Mock training logs
epochs = list(range(1, 11))
train_loss = [2.5, 1.8, 1.3, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1]
val_loss   = [2.6, 1.9, 1.5, 1.2, 1.1, 1.1, 1.2, 1.3, 1.5, 1.7]  # Overfits after epoch 6

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
plt.axvline(x=6, color='g', linestyle='--', label='Optimal Stop (Epoch 6)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch Analysis: Overfitting Detection')
plt.legend()
plt.grid(True)
plt.savefig('learning_lab/experiments/02_evaluation_and_tuning/EX30_Epoch_Analysis/epoch_loss_chart.png')
print("Generalization curve chart saved.")
```

---
*Related Topics:*
*   [[EX29_Batch_Size\|EX29: Batch Size]]
*   [[EX31_Perceptron\|EX31: The Perceptron]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
