import numpy as np

def calculate_pr_curve(y_true, scores):
    """
    Calculate Precision-Recall coordinates from scratch.
    """
    # TODO: Sort prediction scores descending to serve as thresholds
    thresholds = np.zeros(len(scores))
    
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        # TODO: Calculate predictions based on threshold
        y_pred = np.zeros(len(y_true))
        
        # TODO: Calculate TP, FP, FN
        TP = 0
        FP = 0
        FN = 0
        
        # TODO: Compute Precision and Recall, then append to lists
        # Hint: If (TP+FP) is 0, precision is 1.0. Otherwise TP / (TP + FP)
        # Hint: If (TP+FN) is 0, recall is 0.0. Otherwise TP / (TP + FN)
        precision = 1.0
        recall = 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        
    # Append boundary points
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])
    
    return np.array(precisions), np.array(recalls)

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_scores = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.85, 0.4, 0.1, 0.65, 0.3])
    
    prec, rec = calculate_pr_curve(y_true, y_scores)
    
    print("--- Training Results ---")
    if len(prec) > 2 and prec.sum() > len(prec): # Check if non-trivial values exist
        print(f"Number of points in PR: {len(prec)}")
        print(f"Precision values: {np.round(prec, 4)}")
        print(f"Recall values   : {np.round(rec, 4)}")
    else:
        print("PR Curve logic not implemented yet.")
