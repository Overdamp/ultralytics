import numpy as np

def calculate_confusion_matrix(y_true, y_pred, n_classes):
    """
    Calculate raw confusion matrix from scratch.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def calculate_normalized_matrix(cm):
    """
    Calculate normalized confusion matrix.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm.astype(float) / row_sums

if __name__ == "__main__":
    # True labels: 0=flange, 1=valve, 2=gauge
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 1, 1, 2, 0, 2, 2, 1])
    
    cm_raw = calculate_confusion_matrix(y_true, y_pred, n_classes=3)
    cm_norm = calculate_normalized_matrix(cm_raw)
    
    print("--- Training Results ---")
    print("Raw Matrix:\n", cm_raw)
    print("\nNormalized Matrix:\n", np.round(cm_norm, 4))
