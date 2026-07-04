import numpy as np

def calculate_fbeta_score(y_true, y_pred, beta=1.0):
    """
    Calculate F-Beta score from scratch.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    if precision == 0.0 or recall == 0.0:
        return 0.0
        
    num = (1 + beta**2) * precision * recall
    den = (beta**2 * precision) + recall
    return num / den

def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1 score (Beta=1.0).
    """
    return calculate_fbeta_score(y_true, y_pred, beta=1.0)

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    f1 = calculate_f1_score(y_true, y_pred)
    f2 = calculate_fbeta_score(y_true, y_pred, beta=2.0)
    
    print("--- Training Results ---")
    print(f"F1 Score : {f1:.4f} (expected: 0.8000)")
    print(f"F2 Score : {f2:.4f} (expected: 0.8000)")
