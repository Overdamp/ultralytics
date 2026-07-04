import numpy as np

def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.
    
    Args:
        boxes (np.ndarray): Bounding boxes of shape (N, 4) in XYXY format.
        scores (np.ndarray): Confidence scores of shape (N,).
        iou_threshold (float): IoU threshold for overlapping box removal.
        
    Returns:
        list: Indices of the boxes to keep.
    """
    # TODO: Get coordinates of all bounding boxes
    # x1 = ...
    # y1 = ...
    # x2 = ...
    # y2 = ...
    
    # TODO: Calculate area of each bounding box
    # areas = ...
    
    # TODO: Sort bounding boxes by confidence score descending
    # Hint: Use np.argsort() and reverse the order
    # order = ...
    
    keep = []
    
    # TODO: Implement the NMS loop
    # while order.size > 0:
    #     1. Select the index with the highest score: i = order[0]
    #     2. Add index i to keep list
    #     3. For the remaining boxes (order[1:]), calculate intersection coordinates:
    #        xx1 = np.maximum(x1[i], x1[order[1:]])
    #        yy1 = np.maximum(y1[i], y1[order[1:]])
    #        xx2 = np.minimum(x2[i], x2[order[1:]])
    #        yy2 = np.minimum(y2[i], y2[order[1:]])
    #     4. Calculate width, height, and area of intersection
    #     5. Calculate union areas: areas[i] + areas[order[1:]] - intersection
    #     6. Compute IoUs
    #     7. Keep indices where IoU is less than iou_threshold
    #     8. Update order to only keep those indices (remember to shift by 1)
    
    return keep

if __name__ == "__main__":
    # Test cases: boxes in XYXY format
    boxes = np.array([
        [100.0, 100.0, 200.0, 200.0],
        [105.0, 105.0, 205.0, 205.0],  # High overlap with box 0
        [100.0, 100.0, 150.0, 150.0],  # Low overlap with box 0
        [300.0, 300.0, 400.0, 400.0],
        [310.0, 310.0, 410.0, 410.0]   # High overlap with box 3
    ])
    scores = np.array([0.9, 0.8, 0.4, 0.95, 0.85])
    iou_threshold = 0.5
    
    keep_indices = nms(boxes, scores, iou_threshold)
    
    print("--- NMS Results ---")
    if len(keep_indices) > 0:
        print(f"Indices kept: {keep_indices}")
        print(f"Kept Boxes:\n{boxes[keep_indices]}")
    else:
        print("NMS implementation not completed yet.")
