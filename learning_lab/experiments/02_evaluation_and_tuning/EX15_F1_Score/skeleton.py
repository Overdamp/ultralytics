import numpy as np

def calculate_fbeta_score(y_true, y_pred, beta=1.0):
    """
    Calculate F-Beta score from scratch.
    """
    # TODO: Calculate TP, FP, FN
    TP = 0
    FP = 0
    FN = 0
    
    # TODO: Calculate Precision and Recall
    precision = 0.0
    recall = 0.0
    
    if precision == 0.0 or recall == 0.0:
        return 0.0
        
    # TODO: Compute F-Beta score using formula
    # Hint: (1 + beta^2) * P * R / (beta^2 * P + R)
    return 0.0

def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1 score (Beta=1.0).
    """
    # TODO: Fetch F-Beta with beta=1.0
    return 0.0

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    f1 = calculate_f1_score(y_true, y_pred)
    f2 = calculate_fbeta_score(y_true, y_pred, beta=2.0)
    
    print("--- Training Results ---")
    if f1 != 0.0:
        print(f"F1 Score : {f1:.4f} (expected: 0.8000)")
        print(f"F2 Score : {f2:.4f} (expected: 0.8000)")
    else:
        print("F1/F-Beta calculations not implemented yet.")
