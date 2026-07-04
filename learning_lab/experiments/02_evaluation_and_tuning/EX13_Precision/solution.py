import numpy as np

def calculate_precision(y_true, y_pred):
    """
    Calculate Precision: TP / (TP + FP)
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    total_pred_positive = TP + FP
    if total_pred_positive == 0:
        return 1.0
    return TP / total_pred_positive

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    
    precision = calculate_precision(y_true, y_pred)
    
    print("--- Training Results ---")
    print(f"Precision: {precision:.4f} (expected: 0.8000)")
