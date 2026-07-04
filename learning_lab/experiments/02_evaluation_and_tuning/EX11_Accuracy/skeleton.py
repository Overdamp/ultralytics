import numpy as np

def calculate_outcomes(y_true, y_pred):
    """
    Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
    """
    # TODO: Calculate TP, TN, FP, FN using elementwise logic on y_true and y_pred
    # Hint: TP = sum((y_true == 1) & (y_pred == 1))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    return int(TP), int(TN), int(FP), int(FN)

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    """
    # TODO: Fetch outcomes and apply the accuracy formula
    # Hint: (TP + TN) / (TP + TN + FP + FN)
    return 0.0

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    TP, TN, FP, FN = calculate_outcomes(y_true, y_pred)
    accuracy = calculate_accuracy(y_true, y_pred)
    
    print("--- Training Results ---")
    print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
    if accuracy != 0.0:
        print(f"Accuracy: {accuracy:.4f} (expected: 0.8000)")
    else:
        print("Accuracy calculation not implemented yet.")
