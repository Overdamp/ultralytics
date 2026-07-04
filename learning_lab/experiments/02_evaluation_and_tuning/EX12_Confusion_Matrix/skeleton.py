import numpy as np

def calculate_confusion_matrix(y_true, y_pred, n_classes):
    """
    Calculate raw confusion matrix from scratch.
    """
    # TODO: Initialize self-matrix of size n_classes x n_classes with zeros
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # TODO: Loop over elements and increment appropriate cell
    # Hint: cm[true_class, predicted_class] += 1
    return cm

def calculate_normalized_matrix(cm):
    """
    Calculate normalized confusion matrix.
    """
    # TODO: Normalize by actual samples count per class (row sums)
    # Hint: Divide row values by sum of that row
    return cm.astype(float)

if __name__ == "__main__":
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 1, 1, 2, 0, 2, 2, 1])
    
    cm_raw = calculate_confusion_matrix(y_true, y_pred, n_classes=3)
    cm_norm = calculate_normalized_matrix(cm_raw)
    
    print("--- Training Results ---")
    if cm_raw.sum() > 0:
        print("Raw Matrix:\n", cm_raw)
        print("\nNormalized Matrix:\n", np.round(cm_norm, 4))
    else:
        print("Confusion Matrix logic not implemented yet.")
