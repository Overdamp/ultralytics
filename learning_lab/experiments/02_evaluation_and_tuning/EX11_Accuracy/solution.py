import numpy as np

def calculate_outcomes(y_true, y_pred):
    """
    Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return int(TP), int(TN), int(FP), int(FN)

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    """
    TP, TN, FP, FN = calculate_outcomes(y_true, y_pred)
    total = TP + TN + FP + FN
    if total == 0:
        return 0.0
    return (TP + TN) / total

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    TP, TN, FP, FN = calculate_outcomes(y_true, y_pred)
    accuracy = calculate_accuracy(y_true, y_pred)
    
    print("--- Training Results ---")
    print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
    print(f"Accuracy: {accuracy:.4f} (expected: 0.8000)")
