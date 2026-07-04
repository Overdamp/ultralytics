import numpy as np

def calculate_precision(y_true, y_pred):
    """
    Calculate Precision: TP / (TP + FP)
    """
    # TODO: Calculate True Positives (TP) and False Positives (FP)
    # Hint: TP = sum((y_true == 1) & (y_pred == 1))
    # Hint: FP = sum((y_true == 0) & (y_pred == 1))
    TP = 0
    FP = 0
    
    # TODO: Compute Precision
    # Hint: If (TP + FP) is 0, return 1.0. Otherwise return TP / (TP + FP)
    return 0.0

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    precision = calculate_precision(y_true, y_pred)
    
    print("--- Training Results ---")
    if precision != 0.0:
        print(f"Precision: {precision:.4f} (expected: 0.8000)")
    else:
        print("Precision calculation not implemented yet.")
