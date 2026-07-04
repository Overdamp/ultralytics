import numpy as np

# Back up or checkpoint this section of code before starting to modify the large file.
# Note: This is an empty skeleton file being populated.

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    
    Args:
        box1 (list or np.ndarray): Coordinates of the first box [x1, y1, x2, y2].
        box2 (list or np.ndarray): Coordinates of the second box [x1, y1, x2, y2].
        
    Returns:
        float: Intersection over Union (IoU) ratio.
    """
    # TODO: Calculate intersection coordinates
    
    # TODO: Calculate areas of intersection and union
    
    # TODO: Return IoU
    return 0.0

def calculate_ap_single_class(gt_boxes, pred_boxes, iou_threshold=0.5, method="all-point"):
    """
    Calculates Average Precision (AP) for a single class.
    
    Args:
        gt_boxes (list of dict): Ground-truth boxes. Each dict has 'image_id' (int/str) and 'box' [x1, y1, x2, y2].
        pred_boxes (list of dict): Predicted boxes. Each dict has 'image_id', 'box', and 'confidence' (float).
        iou_threshold (float): Minimum IoU threshold to consider a prediction a True Positive.
        method (str): 'all-point' (COCO) or '11-point' (PASCAL VOC 2007) interpolation.
        
    Returns:
        float: Average Precision (AP) value.
        np.ndarray: Cumulative Precisions.
        np.ndarray: Cumulative Recalls.
    """
    # 1. Handle edge cases: No ground-truth boxes or no predicted boxes
    
    # 2. Sort predicted boxes by confidence in descending order
    
    # 3. Track matched ground-truth boxes to prevent double-matching
    # Hint: Create a dictionary matching_tracker mapping image_id to a boolean mask/array of size len(gt_boxes_in_image)
    
    # 4. Initialize arrays for True Positives (TP) and False Positives (FP)
    
    # 5. Determine TP and FP for each prediction
    # Loop over sorted predictions:
    #   - Find all GT boxes in the same image.
    #   - Calculate IoU between the prediction and all GT boxes in that image.
    #   - Find the GT box with the highest IoU.
    #   - If highest IoU >= iou_threshold and that GT box is not yet matched:
    #       - Mark as TP, and mark the GT box as matched.
    #     Else:
    #       - Mark as FP.
    
    # 6. Compute cumulative TP and FP, then cumulative Precision and Recall
    
    # 7. Compute AP using all-point or 11-point interpolation
    # - 11-point interpolation: Average of maximum precision at recall levels [0.0, 0.1, ..., 1.0]
    # - All-point interpolation: Area under the Precision-Recall curve with step-wise interpolation
    
    ap = 0.0
    precisions = np.array([])
    recalls = np.array([])
    
    return ap, precisions, recalls

if __name__ == "__main__":
    # Professor's Toy Example for Single-Class AP Verification:
    # 1 Image, 3 Ground-truth boxes, 5 Predicted boxes.
    
    gt_boxes = [
        {"image_id": 0, "box": [10, 10, 50, 50]},
        {"image_id": 0, "box": [60, 60, 100, 100]},
        {"image_id": 0, "box": [120, 120, 160, 160]}
    ]
    
    pred_boxes = [
        {"image_id": 0, "box": [12, 12, 48, 48], "confidence": 0.95}, # Should match 1st GT (IoU >= 0.5) -> TP
        {"image_id": 0, "box": [58, 58, 98, 98], "confidence": 0.88}, # Should match 2nd GT (IoU >= 0.5) -> TP
        {"image_id": 0, "box": [122, 122, 158, 158], "confidence": 0.75}, # Should match 3rd GT (IoU >= 0.5) -> TP
        {"image_id": 0, "box": [14, 14, 46, 46], "confidence": 0.65}, # Duplicated prediction for 1st GT -> FP
        {"image_id": 0, "box": [200, 200, 240, 240], "confidence": 0.50}  # No matching GT -> FP
    ]
    
    ap, prec, rec = calculate_ap_single_class(gt_boxes, pred_boxes, iou_threshold=0.5, method="all-point")
    print(f"Calculated Single-Class AP: {ap:.4f}")
