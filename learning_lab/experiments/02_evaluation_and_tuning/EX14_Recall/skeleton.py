import numpy as np

def calculate_recall(y_true, y_pred):
    """
    Calculate Recall: TP / (TP + FN)
    """
    # TODO: Calculate True Positives (TP) and False Negatives (FN)
    # Hint: TP = sum((y_true == 1) & (y_pred == 1))
    # Hint: FN = sum((y_true == 1) & (y_pred == 0))
    TP = 0
    FN = 0
    
    # TODO: Compute Recall
    # Hint: If (TP + FN) is 0, return 1.0. Otherwise return TP / (TP + FN)
    return 0.0

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    recall = calculate_recall(y_true, y_pred)
    
    print("--- Training Results ---")
    if recall != 0.0:
        print(f"Recall: {recall:.4f} (expected: 0.8000)")
    else:
        print("Recall calculation not implemented yet.")
