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
    if len(boxes) == 0:
        return []
        
    # 1. Get coordinates of all bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 2. Calculate area of each bounding box
    areas = (x2 - x1) * (y2 - y1)
    
    # 3. Sort bounding boxes by confidence score descending
    order = scores.argsort()[::-1]
    
    keep = []
    
    # 4. NMS loop
    while order.size > 0:
        # Select the index with the highest score
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # For the remaining boxes, calculate intersection coordinates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Calculate overlap dimensions and area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        # Compute IoU
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Keep boxes where IoU is less than iou_threshold
        inds = np.where(iou <= iou_threshold)[0]
        # Update order (shift by 1 to skip order[0])
        order = order[inds + 1]
        
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
    print(f"Indices kept: {[int(x) for x in keep_indices]} (Expected: [3, 0, 2])")
    print(f"Kept Boxes:\n{boxes[keep_indices]}")
