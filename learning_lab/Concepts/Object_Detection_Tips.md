# Object Detection Tips & Best Practices

Tips for training state-of-the-art YOLO models on your custom datasets:

## 1. Data Quality Checks
*   **Resolution:** Train with an image size (`imgsz`) close to the native camera resolution or target deployment resolution (usually `640` or `1280`).
*   **Agnostic Classes:** Ensure there is no class imbalance (e.g. 5000 images of `flange` and only 10 images of `pig-alert`).
*   **Verify Labels:** Use visualization tools to ensure bounding boxes are tight and correctly centered.

## 2. Preventing Overfitting
*   If validation loss starts to rise while training loss falls, the model is overfitting.
*   **Solutions:**
    *   Increase augmentations (`mosaic`, `mixup`).
    *   Increase weight decay.
    *   Stop training early using patience parameters.

---
*Related Topics:*
*   [[YOLO_Loss_Functions]]
*   Back to [[learning_journal]]
