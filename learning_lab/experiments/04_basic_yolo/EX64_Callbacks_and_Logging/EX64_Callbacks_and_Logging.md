# 🧠 EX64: Callbacks and Logging

During training, monitoring metrics like loss, precision, recall, and mAP is essential for understanding model convergence. Ultralytics YOLO has built-in integration with popular logging platforms and provides a callback system to execute custom code at specific training events.

---

## 1. Architectural Design: The Observer Pattern

Under the hood, the YOLO training engine manages execution events using a **Publisher-Subscriber (Observer) Pattern**. The core training loop triggers event hooks at critical execution boundaries. Any registered callback function (subscriber) is called dynamically, receiving the current `trainer`, `validator`, or `predictor` instance as a parameter.

```mermaid
sequenceDiagram
    participant loop as Training Loop
    participant manager as Callback Manager
    participant sub as Registered Callbacks
    
    loop->>manager: trigger("on_train_start")
    manager->>sub: execute with trainer state
    loop->>loop: Run Epoch
    loop->>manager: trigger("on_train_epoch_end")
    manager->>sub: execute with epoch metrics
    loop->>manager: trigger("on_fit_epoch_end")
    manager->>sub: execute with validation results
```

This decoupling allows developers to inject logging logic, early stopping conditions, or hardware telemetry without altering the underlying training code.

---

## 2. Comprehensive Callback Event Reference

YOLO exposes a rich set of hooks across the training lifecycle:

| Event Hook | Trigger Location | Common Use Case |
| :--- | :--- | :--- |
| `on_pretrain_routine_start` | Before dataset checks and directory setup. | Initializing remote workspace managers or clearing previous cached files. |
| `on_train_start` | Right before the first training batch starts. | Starting timers, printing configuration tables, logging initial weights structure. |
| `on_train_epoch_start` | At the beginning of each epoch loop. | Resetting epoch-specific logs or preparing custom data splits. |
| `on_train_batch_end` | After backpropagation updates of a batch. | Real-time batch loss logging or step adjustments. |
| `on_train_epoch_end` | After processing all training batches. | Tracking training loss items and learning rates. |
| `on_fit_epoch_end` | After the validation evaluation finishes. | Extracting mAP metrics, evaluating early stopping, saving custom checkpoints. |
| `on_train_end` | Once the total epochs are completed. | final model serialization, cloud log synchronization, training report compilation. |

---

## 3. Python API vs. CLI: Callback Capabilities

*   **Python API**: Fully supports callback registration. You can bind custom Python functions to any hook using `model.add_callback()`.
*   **CLI**: Cannot register custom runtime code hooks. You are limited to the default automated logging backends (TensorBoard, MLflow, Weights & Biases) that initialize based on installed packages.

---

## 4. Python Code Demonstration

Here is how you write a custom callback to monitor epoch loss and validation metrics, register it, and start training.

```python
import torch
from ultralytics import YOLO

# 1. Define custom callback function
def log_epoch_summary(trainer):
    """
    A custom callback triggered at the end of each fit epoch.
    Prints a customized message showing loss and fitness metrics.
    """
    epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
    total_epochs = trainer.epochs
    
    # Extract loss values
    loss_val = trainer.loss_items.tolist()
    
    # Extract validation metrics
    metrics = trainer.metrics
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    
    print(f"\n=========================================")
    print(f"🎓 Professor Monitor - Epoch [{epoch}/{total_epochs}]")
    print(f"--> Current Box/Class/DFL Losses: {loss_val}")
    print(f"--> Validation mAP50-95: {map50_95:.4f}")
    print(f"=========================================\n")

# 2. Load the model
model = YOLO("yolo11n.pt")

# 3. Register the callback to the 'on_fit_epoch_end' hook
model.add_callback("on_fit_epoch_end", log_epoch_summary)

# 4. Start training
model.train(
    data="coco8.yaml",
    epochs=3,
    imgsz=640,
    device="cpu"
)
```

---

## 💡 Professor Tips

### 1. Cleaning and Overwriting Default Callbacks
YOLO registers default logger callbacks (e.g. TensorBoard and W&B) on startup. If you want to disable default loggers to prevent duplicate logging or speed up execution, clear the registered lists:
```python
# Clear all default callbacks for fit epoch end
model.callbacks["on_fit_epoch_end"] = []

# Register only your custom callback
model.add_callback("on_fit_epoch_end", log_epoch_summary)
```

### 2. Telemetry and VRAM Monitoring inside Callbacks
You can check GPU temperatures and memory allocation inside training hooks to pause training if limits are exceeded, preventing system crashes:
```python
def monitor_gpu_vram(trainer):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"⚡ VRAM: {allocated:.1f}MB (Max: {max_allocated:.1f}MB)")

model.add_callback("on_train_epoch_end", monitor_gpu_vram)
```

---
*Related Topics:*
*   [[EX53_Training_Settings|EX53: Training Settings]]
*   [[EX55_Evaluation_Modes|EX55: Evaluation & Validation Modes]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
