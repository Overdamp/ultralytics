import numpy as np

def calculate_pr_curve(y_true, scores):
    """
    Calculate Precision-Recall coordinates from scratch.
    """
    # Sort scores descending to use as thresholds
    thresholds = np.sort(scores)[::-1]
    
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
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
    print(f"Number of points in PR: {len(prec)}")
    print(f"Precision values: {np.round(prec, 4)}")
    print(f"Recall values   : {np.round(rec, 4)}")
