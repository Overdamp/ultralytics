import numpy as np

def calculate_roc_curve(y_true, scores):
    """
    Calculate ROC curve coordinates (FPR, TPR) and thresholds from scratch.
    """
    # TODO: Sort prediction scores in descending order to serve as thresholds
    # Hint: Use np.sort(scores)[::-1] and prepend a value slightly greater than 1.0
    thresholds = np.zeros(len(scores))
    
    tprs = []
    fprs = []
    
    for thresh in thresholds:
        # TODO: Calculate predictions based on threshold
        y_pred = np.zeros(len(y_true))
        
        # TODO: Calculate TP, TN, FP, FN
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        # TODO: Compute TPR and FPR, then append to tprs and fprs
        tpr = 0.0
        fpr = 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
    return np.array(fprs), np.array(tprs), thresholds

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_scores = np.array([0.9, 0.1, 0.8, 0.75, 0.2, 0.85, 0.4, 0.15, 0.7, 0.3])
    
    fpr, tpr, thresholds = calculate_roc_curve(y_true, y_scores)
    
    print("--- Training Results ---")
    if len(fpr) > 0 and fpr.sum() > 0:
        print(f"Number of points in ROC: {len(fpr)}")
        print(f"FPR values: {np.round(fpr, 4)}")
        print(f"TPR values: {np.round(tpr, 4)}")
    else:
        print("ROC curve logic not implemented yet.")
