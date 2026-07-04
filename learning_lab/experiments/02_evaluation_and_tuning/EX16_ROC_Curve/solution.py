import numpy as np

def calculate_roc_curve(y_true, scores):
    """
    Calculate ROC curve coordinates (FPR, TPR) and thresholds from scratch.
    """
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
        
    return np.array(fprs), np.array(tprs), thresholds

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_scores = np.array([0.9, 0.1, 0.8, 0.75, 0.2, 0.85, 0.4, 0.15, 0.7, 0.3])
    
    fpr, tpr, thresholds = calculate_roc_curve(y_true, y_scores)
    
    print("--- Training Results ---")
    print(f"Number of points in ROC: {len(fpr)}")
    print(f"FPR values: {np.round(fpr, 4)}")
    print(f"TPR values: {np.round(tpr, 4)}")
