import numpy as np

# Back up or checkpoint this section of code before starting to modify the large file.
# Note: Developing the full solution file for Mean Average Precision (mAP) calculation.

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
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area

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
    num_gts = len(gt_boxes)
    num_preds = len(pred_boxes)
    
    # Handle edge cases
    if num_gts == 0:
        return 0.0, np.zeros(num_preds), np.zeros(num_preds)
    if num_preds == 0:
        return 0.0, np.array([]), np.array([])
        
    # Sort predictions by confidence score descending
    sorted_preds = sorted(pred_boxes, key=lambda x: x["confidence"], reverse=True)
    
    # Group ground truths by image_id
    gt_by_img = {}
    for gt in gt_boxes:
        img_id = gt["image_id"]
        if img_id not in gt_by_img:
            gt_by_img[img_id] = []
        gt_by_img[img_id].append(gt)
        
    # Track matched ground truths to prevent double-matching
    gt_matched = {}
    for img_id, boxes in gt_by_img.items():
        gt_matched[img_id] = [False] * len(boxes)
        
    tps = np.zeros(num_preds)
    fps = np.zeros(num_preds)
    
    # Determine TP/FP for each prediction
    for idx, pred in enumerate(sorted_preds):
        img_id = pred["image_id"]
        pred_box = pred["box"]
        
        # If no ground-truth box exists in this image, it's a False Positive
        if img_id not in gt_by_img or len(gt_by_img[img_id]) == 0:
            fps[idx] = 1.0
            continue
            
        best_iou = -1.0
        best_gt_idx = -1
        
        # Calculate IoU with all ground-truth boxes in the same image
        for gt_idx, gt in enumerate(gt_by_img[img_id]):
            iou = calculate_iou(pred_box, gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        # Match if best IoU exceeds threshold and the GT box is not yet matched
        if best_iou >= iou_threshold:
            if not gt_matched[img_id][best_gt_idx]:
                tps[idx] = 1.0
                gt_matched[img_id][best_gt_idx] = True
            else:
                # GT already matched to a prediction with higher confidence -> False Positive
                fps[idx] = 1.0
        else:
            # Low overlap -> False Positive
            fps[idx] = 1.0
            
    # Compute cumulative sums of TP and FP
    cum_tps = np.cumsum(tps)
    cum_fps = np.cumsum(fps)
    
    # Calculate Precision and Recall
    # Precision = TP / (TP + FP)
    # Recall = TP / Num Ground Truths
    precisions = cum_tps / (cum_tps + cum_fps)
    recalls = cum_tps / num_gts
    
    ap = 0.0
    if method == "11-point":
        # 11-point interpolation: 0.0, 0.1, ..., 1.0
        for t in np.linspace(0.0, 1.0, 11):
            matching_indices = np.where(recalls >= t)[0]
            p_at_t = np.max(precisions[matching_indices]) if len(matching_indices) > 0 else 0.0
            ap += p_at_t / 11.0
    else:
        # All-point interpolation (COCO standard)
        # Prepend 0.0 and append 1.0 to recall, prepend 0.0 and append 0.0 to precision
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        
        # Compute the running maximum precision from right to left
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
            
        # Find where the recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # AP is the sum of rectangle areas under the step curve
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
    return ap, precisions, recalls

def calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5], method="all-point"):
    """
    Calculates Mean Average Precision (mAP) across multiple classes and IoU thresholds.
    
    Args:
        gt_boxes (list of dict): Ground-truth boxes with keys 'image_id', 'class_id', 'box'.
        pred_boxes (list of dict): Predicted boxes with keys 'image_id', 'class_id', 'box', 'confidence'.
        iou_thresholds (list of float): IoU thresholds to average over.
        method (str): 'all-point' or '11-point' interpolation.
        
    Returns:
        dict: Containing:
            - 'class_ap': Dict mapping class_id -> list of AP values (one per IoU threshold)
            - 'map_at_thresholds': Dict mapping iou_threshold -> mAP at that threshold
            - 'overall_map': Scalar representing final averaged mAP
    """
    # Find unique classes across both ground-truths and predictions
    classes = np.unique(
        [gt["class_id"] for gt in gt_boxes] + [pred["class_id"] for pred in pred_boxes]
    )
    
    class_ap = {cid: [] for cid in classes}
    map_at_thresholds = {}
    
    for iou_thresh in iou_thresholds:
        aps = []
        for cid in classes:
            # Filter GT and Predictions for the current class
            class_gts = [gt for gt in gt_boxes if gt["class_id"] == cid]
            class_preds = [pred for pred in pred_boxes if pred["class_id"] == cid]
            
            ap, _, _ = calculate_ap_single_class(
                class_gts, class_preds, iou_threshold=iou_thresh, method=method
            )
            class_ap[cid].append(ap)
            aps.append(ap)
            
        map_at_thresholds[iou_thresh] = np.mean(aps) if len(aps) > 0 else 0.0
        
    overall_map = np.mean(list(map_at_thresholds.values()))
    
    return {
        "class_ap": class_ap,
        "map_at_thresholds": map_at_thresholds,
        "overall_map": overall_map
    }

if __name__ == "__main__":
    # --- Part 1: Professor's Single-Class AP Verification ---
    print("=== Professor's Single-Class AP Verification ===")
    gt_boxes_single = [
        {"image_id": 0, "box": [10, 10, 50, 50]},
        {"image_id": 0, "box": [60, 60, 100, 100]},
        {"image_id": 0, "box": [120, 120, 160, 160]}
    ]
    
    pred_boxes_single = [
        {"image_id": 0, "box": [12, 12, 48, 48], "confidence": 0.95}, # TP (matches GT 1)
        {"image_id": 0, "box": [58, 58, 98, 98], "confidence": 0.88}, # TP (matches GT 2)
        {"image_id": 0, "box": [122, 122, 158, 158], "confidence": 0.75}, # TP (matches GT 3)
        {"image_id": 0, "box": [14, 14, 46, 46], "confidence": 0.65}, # FP (Duplicate of GT 1)
        {"image_id": 0, "box": [200, 200, 240, 240], "confidence": 0.50}  # FP (No overlap)
    ]
    
    ap_all, prec_all, rec_all = calculate_ap_single_class(
        gt_boxes_single, pred_boxes_single, iou_threshold=0.5, method="all-point"
    )
    ap_11, prec_11, rec_11 = calculate_ap_single_class(
        gt_boxes_single, pred_boxes_single, iou_threshold=0.5, method="11-point"
    )
    
    print(f"Sorted Confidence Predictions:")
    for idx, p in enumerate(sorted(pred_boxes_single, key=lambda x: x["confidence"], reverse=True)):
        print(f"  Pred {idx+1}: Conf={p['confidence']:.2f} | Precision={prec_all[idx]:.4f} | Recall={rec_all[idx]:.4f}")
    
    print(f"\nAll-point Interpolation AP: {ap_all:.4f} (Expected: 1.0000)")
    print(f"11-point Interpolation AP: {ap_11:.4f} (Expected: 1.0000)")
    
    # --- Part 2: Multiclass Evaluation on Mock Data ---
    print("\n=== Multiclass Evaluation on Mock Data ===")
    
    # Let's define mock data for three classes:
    # 0: 'control-valve'
    # 1: 'small-valve'
    # 2: 'flange'
    # Across 2 images:
    mock_gts = [
        # Image 0
        {"image_id": 0, "class_id": 0, "box": [10, 10, 50, 50]},
        {"image_id": 0, "class_id": 1, "box": [60, 60, 100, 100]},
        {"image_id": 0, "class_id": 2, "box": [120, 120, 160, 160]},
        # Image 1
        {"image_id": 1, "class_id": 0, "box": [20, 20, 70, 70]},
        {"image_id": 1, "class_id": 2, "box": [80, 80, 140, 140]}
    ]
    
    mock_preds = [
        # Image 0
        {"image_id": 0, "class_id": 0, "box": [12, 12, 48, 48], "confidence": 0.95},  # TP for class 0
        {"image_id": 0, "class_id": 1, "box": [58, 58, 95, 95], "confidence": 0.85},   # TP for class 1
        {"image_id": 0, "class_id": 2, "box": [122, 122, 158, 158], "confidence": 0.90}, # TP for class 2 (high IoU match)
        {"image_id": 0, "class_id": 0, "box": [14, 14, 46, 46], "confidence": 0.60},    # FP for class 0 (duplicate)
        # Image 1
        {"image_id": 1, "class_id": 0, "box": [25, 25, 68, 68], "confidence": 0.92},   # TP for class 0
        {"image_id": 1, "class_id": 1, "box": [10, 80, 40, 110], "confidence": 0.45},   # FP for class 1 (false alarm)
        {"image_id": 1, "class_id": 2, "box": [82, 82, 138, 138], "confidence": 0.70}  # TP for class 2 (matches GT with good IoU)
    ]
    
    # Calculate mAP@0.5
    res_50 = calculate_map(mock_gts, mock_preds, iou_thresholds=[0.5], method="all-point")
    print(f"\nmAP@0.5 results:")
    for cid, ap in res_50["class_ap"].items():
        class_name = {0: "control-valve", 1: "small-valve", 2: "flange"}[cid]
        print(f"  Class '{class_name}' AP@0.5: {ap[0]:.4f}")
    print(f"  Overall mAP@0.5: {res_50['overall_map']:.4f}")
    
    # Calculate mAP@0.5:0.95
    thresholds = np.linspace(0.5, 0.95, 10)
    res_coco = calculate_map(mock_gts, mock_preds, iou_thresholds=thresholds, method="all-point")
    print(f"\nmAP@0.5:0.95 results:")
    for cid, aps in res_coco["class_ap"].items():
        class_name = {0: "control-valve", 1: "small-valve", 2: "flange"}[cid]
        mean_ap = np.mean(aps)
        print(f"  Class '{class_name}' AP@0.5:0.95: {mean_ap:.4f}")
    print(f"  Overall mAP@0.5:0.95 (COCO standard): {res_coco['overall_map']:.4f}")
