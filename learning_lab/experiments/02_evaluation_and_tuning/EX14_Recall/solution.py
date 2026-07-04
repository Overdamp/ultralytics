import numpy as np

def calculate_recall(y_true, y_pred):
    """
    Calculate Recall: TP / (TP + FN)
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    actual_positives = TP + FN
    if actual_positives == 0:
        return 1.0
    return TP / actual_positives

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    recall = calculate_recall(y_true, y_pred)
    
    print("--- Training Results ---")
    print(f"Recall: {recall:.4f} (expected: 0.8000)")
