import numpy as np

# Back up or checkpoint this section of code before starting to modify the large file.
# Note: This is an empty solution file being populated.

def calculate_classic_confidence(p_class_given_obj, p_obj, iou):
    """
    Calculates the classic YOLO (v1-v5) class-specific confidence score.
    
    Formula:
        Class Score = P(Class_i | Object) * P(Object) * IoU
        
    Args:
        p_class_given_obj (float or np.ndarray): Conditional probability of class given object.
        p_obj (float or np.ndarray): Objectness score (probability that cell contains an object).
        iou (float or np.ndarray): Intersection over Union between predicted and ground-truth box.
        
    Returns:
        float or np.ndarray: Classic class-specific confidence score.
    """
    return p_class_given_obj * p_obj * iou

def calculate_task_aligned_score(s, iou, alpha=0.5, beta=6.0):
    """
    Calculates the Task-Aligned Assigner score used in anchor-free YOLO (v8-v11).
    
    Formula:
        t = s^alpha * IoU^beta
        
    Args:
        s (float or np.ndarray): Predicted class probability (classification score).
        iou (float or np.ndarray): Intersection over Union between predicted and ground-truth box.
        alpha (float): Power weight for the classification score. Default is 0.5.
        beta (float): Power weight for the localization score (IoU). Default is 6.0.
        
    Returns:
        float or np.ndarray: Task-aligned score.
    """
    return (s ** alpha) * (iou ** beta)

def evaluate_predictions(predictions, num_ground_truths, conf_threshold, iou_threshold=0.5):
    """
    Evaluates predictions against ground truth for a given confidence threshold.
    
    Args:
        predictions (list of dict): List of dicts, each with keys 'score' (float) and 'iou' (float).
                                     To simplify, assume each prediction is matched to the best ground truth,
                                     and 'iou' represents the IoU with that ground truth.
        num_ground_truths (int): Total number of ground-truth objects in the dataset.
        conf_threshold (float): Bounding box confidence score threshold.
        iou_threshold (float): IoU threshold to consider a detection as a True Positive (TP).
        
    Returns:
        dict: Containing 'TP', 'FP', 'FN', 'Precision', and 'Recall'.
    """
    filtered_preds = [p for p in predictions if p['score'] >= conf_threshold]
    
    tp = sum(1 for p in filtered_preds if p['iou'] >= iou_threshold)
    fp = sum(1 for p in filtered_preds if p['iou'] < iou_threshold)
    fn = num_ground_truths - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / num_ground_truths if num_ground_truths > 0 else 0.0
    
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall
    }

if __name__ == "__main__":
    # 1. Verification of mathematical calculations
    p_cls = 0.9
    p_obj = 0.8
    iou = 0.75
    
    classic_score = calculate_classic_confidence(p_cls, p_obj, iou)
    print(f"Classic Confidence: {classic_score:.4f} (Expected: 0.5400)")
    
    ta_score = calculate_task_aligned_score(p_cls, iou, alpha=0.5, beta=6.0)
    print(f"Task-Aligned Score: {ta_score:.4f} (Expected: 0.1691)")
    
    # 2. Test predictions dataset
    # We simulate 10 ground-truth objects
    num_gt = 10
    
    # Let's define some mock predictions with different scores and IoU overlaps
    mock_predictions = [
        {"score": 0.95, "iou": 0.85},  # TP
        {"score": 0.88, "iou": 0.92},  # TP
        {"score": 0.75, "iou": 0.40},  # FP (low IoU)
        {"score": 0.65, "iou": 0.70},  # TP
        {"score": 0.55, "iou": 0.80},  # TP
        {"score": 0.45, "iou": 0.30},  # FP (low IoU)
        {"score": 0.35, "iou": 0.65},  # TP
        {"score": 0.20, "iou": 0.15},  # FP (low IoU)
        {"score": 0.12, "iou": 0.55},  # TP (but low score)
    ]
    
    print("\n--- Precision-Recall Evaluation at Different Thresholds ---")
    for thresh in [0.15, 0.30, 0.50, 0.70, 0.90]:
        metrics = evaluate_predictions(mock_predictions, num_gt, thresh)
        print(f"Threshold: {thresh:.2f} | TP: {metrics['TP']} | FP: {metrics['FP']} | FN: {metrics['FN']} | "
              f"Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f}")
