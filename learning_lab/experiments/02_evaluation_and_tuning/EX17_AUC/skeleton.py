import numpy as np

def calculate_auc_score(y_true, scores):
    """
    Calculate AUC score from scratch using trapezoidal integration.
    """
    # 1. Calculate ROC Curve coordinates
    thresholds = np.sort(scores)[::-1]
    thresholds = np.concatenate([[1.001], thresholds])
    
    tprs = []
    fprs = []
    
    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (TN + FP) if (TN + FP) > 0 else 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
    # 2. TODO: Integrate using Trapezoidal Rule
    # Hint: Loop over fprs and tprs, computing sum of 0.5 * (tprs[i] + tprs[i-1]) * (fprs[i] - fprs[i-1])
    area = 0.0
    return area

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_scores = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.85, 0.4, 0.1, 0.65, 0.3])
    
    auc_val = calculate_auc_score(y_true, y_scores)
    
    print("--- Training Results ---")
    if auc_val != 0.0:
        print(f"AUC Score: {auc_val:.4f} (expected: 1.0000)")
    else:
        print("AUC integration logic not implemented yet.")
