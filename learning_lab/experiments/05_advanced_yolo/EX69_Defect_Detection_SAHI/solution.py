import numpy as np
from typing import List, Tuple, Union, Any
from ultralytics import YOLO

def sahi_inference(
    image: np.ndarray,
    slice_size: Union[int, Tuple[int, int]],
    overlap_ratio: float,
    model: YOLO,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45
) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
    """
    Performs Sliced Aided Hyper Inference (SAHI) on high-resolution images.
    
    This function:
    1. Slices the input high-resolution image into smaller, overlapping windows of slice_size.
    2. Runs standard YOLO object detection on each window (slice).
    3. Translates detected box coordinates from slice local coordinates back to global image coordinates.
    4. Merges all predictions across all slices.
    5. Performs Non-Maximum Suppression (NMS) on global predictions to remove duplicate boxes
       created by slice overlap.
       
    Args:
        image (np.ndarray): High-resolution input image.
        slice_size (Union[int, Tuple[int, int]]): Height and width of the slices.
            Can be an integer (e.g. 640) or a tuple (slice_height, slice_width).
        overlap_ratio (float): Fraction of overlap between adjacent slices (between 0.0 and 1.0).
        model (YOLO): The YOLO model instance to run inference on each slice.
        conf_threshold (float): Confidence threshold for YOLO detection.
        nms_threshold (float): Intersection-over-Union (IoU) threshold for NMS.
        
    Returns:
        List[Tuple[Tuple[float, float, float, float], float, int]]: List of merged, final detections.
            Each detection is a tuple of:
            - Bounding box in global coordinates: (x1, y1, x2, y2)
            - Confidence score: float
            - Class ID: int
    """
    if image is None or image.size == 0:
        return []
        
    h_img, w_img, _ = image.shape
    
    # 1. Parse slice sizes
    if isinstance(slice_size, int):
        slice_h = slice_w = slice_size
    else:
        slice_h, slice_w = slice_size
        
    # Limit slice sizes to image dimensions if image is smaller than slice_size
    slice_h = min(slice_h, h_img)
    slice_w = min(slice_w, w_img)
    
    # 2. Calculate steps
    step_h = int(slice_h * (1.0 - overlap_ratio))
    step_w = int(slice_w * (1.0 - overlap_ratio))
    
    # Ensure step sizes are at least 1 pixel
    step_h = max(1, step_h)
    step_w = max(1, step_w)
    
    # 3. Generate slice start coordinates
    y_starts = []
    y = 0
    while y + slice_h <= h_img:
        y_starts.append(y)
        y += step_h
    # Ensure the bottom edge is covered
    if not y_starts or y_starts[-1] + slice_h < h_img:
        y_starts.append(h_img - slice_h)
    y_starts = sorted(list(set(y_starts)))
    
    x_starts = []
    x = 0
    while x + slice_w <= w_img:
        x_starts.append(x)
        x += step_w
    # Ensure the right edge is covered
    if not x_starts or x_starts[-1] + slice_w < w_img:
        x_starts.append(w_img - slice_w)
    x_starts = sorted(list(set(x_starts)))
    
    # 4. Perform Slices Inference and Coordinate Translation
    global_boxes = []
    global_confs = []
    global_classes = []
    
    for y_start in y_starts:
        for x_start in x_starts:
            # Crop the window
            y_end = y_start + slice_h
            x_end = x_start + slice_w
            crop = image[y_start:y_end, x_start:x_end]
            
            if crop.size == 0:
                continue
                
            # Run YOLO inference
            results = model(crop, conf=conf_threshold, verbose=False)[0]
            
            # Translate coordinates back to global frame
            for box in results.boxes:
                # Bbox coordinates on crop
                x1_c, y1_c, x2_c, y2_c = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Global coordinates
                x1_g = x1_c + x_start
                y1_g = y1_c + y_start
                x2_g = x2_c + x_start
                y2_g = y2_c + y_start
                
                global_boxes.append([x1_g, y1_g, x2_g, y2_g])
                global_confs.append(conf)
                global_classes.append(cls)
                
    if not global_boxes:
        return []
        
    # Convert lists to NumPy arrays for faster NMS processing
    boxes = np.array(global_boxes)
    confs = np.array(global_confs)
    classes = np.array(global_classes)
    
    # 5. Non-Maximum Suppression (NMS)
    # Sort detections by confidence score in descending order
    order = np.argsort(confs)[::-1]
    
    keep_indices = []
    
    while len(order) > 0:
        idx = order[0]
        keep_indices.append(idx)
        
        if len(order) == 1:
            break
            
        # Compute IoU between the current box and the remaining boxes
        curr_box = boxes[idx]
        rem_boxes = boxes[order[1:]]
        rem_classes = classes[order[1:]]
        
        # Calculate intersection
        ix1 = np.maximum(curr_box[0], rem_boxes[:, 0])
        iy1 = np.maximum(curr_box[1], rem_boxes[:, 1])
        ix2 = np.minimum(curr_box[2], rem_boxes[:, 2])
        iy2 = np.minimum(curr_box[3], rem_boxes[:, 3])
        
        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        intersection = inter_w * inter_h
        
        # Calculate union
        curr_area = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
        rem_areas = (rem_boxes[:, 2] - rem_boxes[:, 0]) * (rem_boxes[:, 3] - rem_boxes[:, 1])
        union = curr_area + rem_areas - intersection
        
        # Avoid division by zero
        iou = np.zeros_like(intersection)
        valid_union = union > 0
        iou[valid_union] = intersection[valid_union] / union[valid_union]
        
        # Multiclass NMS: Suppress if IoU > nms_threshold AND class is the same
        same_class = rem_classes == classes[idx]
        suppressed = (iou > nms_threshold) & same_class
        
        # Keep only the boxes that are not suppressed
        order = order[1:][~suppressed]
        
    # Format the final detections
    final_detections = []
    for k_idx in keep_indices:
        box_tuple = tuple(boxes[k_idx])
        final_detections.append((
            (box_tuple[0], box_tuple[1], box_tuple[2], box_tuple[3]),
            float(confs[k_idx]),
            int(classes[k_idx])
        ))
        
    return final_detections
