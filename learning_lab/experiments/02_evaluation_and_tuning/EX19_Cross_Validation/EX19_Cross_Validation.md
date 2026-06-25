# 🧠 EX19: Cross-Validation

Cross-Validation is a resampling technique used to evaluate model generalization on a limited dataset. It helps prevent overfitting and ensures the validation metrics are not biased by a single, lucky train-test split.

---

## 1. K-Fold Cross-Validation Step-by-Step

In $K$-Fold Cross-Validation, the dataset is split into $K$ equal-sized subsets (folds):

```
Dataset Folds:
Epoch 1:  [  VAL  ] [ Train ] [ Train ] [ Train ] [ Train ] -> mAP_1
Epoch 2:  [ Train ] [  VAL  ] [ Train ] [ Train ] [ Train ] -> mAP_2
Epoch 3:  [ Train ] [ Train ] [  VAL  ] [ Train ] [ Train ] -> mAP_3
...
```

1.  Split the data randomly into $K$ subsets (typically $K=5$ or $K=10$).
2.  Train the model on $K-1$ subsets combined.
3.  Evaluate the model on the remaining 1 subset (validation set).
4.  Repeat this process $K$ times, using a different subset as the validation set each time.
5.  Average the $K$ evaluation scores to obtain the final performance metric:

$$\text{mAP}_{\text{final}} = \frac{1}{K} \sum_{i=1}^{K} \text{mAP}_i$$

---

## 2. Stratified K-Fold (Classification Standard)
When dealing with class imbalance (e.g. if one class represents only $2\%$ of your dataset), a random split might end up with folds that contain zero samples of that rare class. 

**Stratified K-Fold** ensures that each fold contains approximately the same percentage of samples of each class as the complete dataset, preserving representation during training.

---

## 🔬 Computer Vision Case Study: K-Fold in YOLO
In deep learning (especially object detection), full $K$-Fold cross-validation is **rarely used for large datasets** because it increases training time and compute cost by $K$ times (e.g. training 5 separate models instead of 1).

However, it is highly recommended when:
1.  **Small Datasets:** You have a small set of custom images and want to prevent bias.
2.  **Hyperparameter Search:** Finding the most robust parameters before starting final training runs.

### Python K-Fold Dataset Splitting Script:
Below is how we partition images and labels into folds using `sklearn`:

```python
from sklearn.model_selection import KFold
import glob
import os

# Get all images in custom dataset
image_files = sorted(glob.glob('datasets/coco8/overall-ptt-object-detection.v11i.yolov11/train/images/*.jpg'))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
    print(f"--- Fold {fold + 1} ---")
    print(f"Number of training samples: {len(train_idx)}")
    print(f"Number of validation samples: {len(val_idx)}")
    # In practice, you copy these files to folders/fold_i and generate data_fold_i.yaml
```

---
*Related Topics:*
*   [[EX18_Precision_Recall_Curve\|EX18: Precision-Recall Curve]]
*   [[EX20_Bias_Variance\|EX20: Bias-Variance Tradeoff]]
*   Return to main plan: [[YOLO_Learning_Plan]]
*   Progress log: [[learning_journal]]
