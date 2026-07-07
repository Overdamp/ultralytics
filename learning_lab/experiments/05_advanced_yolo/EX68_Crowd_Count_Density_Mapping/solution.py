import numpy as np
from typing import List, Tuple, Any

def estimate_crowd_size(
    image: np.ndarray,
    sparse_boxes: List[Tuple[float, float, float, float]],
    dense_crop_regions: List[Tuple[float, float, float, float]],
    density_model: Any,
    overlap_threshold: float = 0.3
) -> float:
    """
    Estimates the total crowd size in an image using a hybrid object detection and density estimation approach.
    
    To prevent double-counting, this function:
    1. Filters the YOLO bounding boxes (`sparse_boxes`) to keep only those that do NOT overlap
       significantly with the designated high-density regions (`dense_crop_regions`).
    2. Sums the remaining YOLO boxes to count people in the sparse regions.
    3. Crops each high-density region from the image, feeds it to a density estimation model
       (which returns a 2D density map), and sums (integrates) the values of the density map
       to get the crowd count in each dense region.
    4. Computes and returns the sum of the sparse count and dense counts.
    
    Args:
        image (np.ndarray): The input image.
        sparse_boxes (List[Tuple[float, float, float, float]]): List of YOLO bounding boxes
            represented as [x1, y1, x2, y2].
        dense_crop_regions (List[Tuple[float, float, float, float]]): List of bounding boxes
            defining congested crowd zones as [x1, y1, x2, y2].
        density_model (Any): Density estimation model. Calling `density_model(crop_image)`
            returns a 2D numpy array (density map) representing person density.
        overlap_threshold (float): Threshold of intersection area ratio above which
            a sparse box is discarded (considered inside a dense region).
            
    Returns:
        float: The estimated total crowd size.
    """
    filtered_sparse_count = 0
    
    # 1. Filter sparse boxes to prevent double counting
    for box in sparse_boxes:
        sx1, sy1, sx2, sy2 = box
        s_area = (sx2 - sx1) * (sy2 - sy1)
        if s_area <= 0:
            continue
            
        max_overlap_ratio = 0.0
        for r_box in dense_crop_regions:
            rx1, ry1, rx2, ry2 = r_box
            
            # Intersection coordinates
            ix1 = max(sx1, rx1)
            iy1 = max(sy1, ry1)
            ix2 = min(sx2, rx2)
            iy2 = min(sy2, ry2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = intersection_area / s_area
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    
        # Keep the box if its overlap with dense regions is below the threshold
        if max_overlap_ratio <= overlap_threshold:
            filtered_sparse_count += 1
            
    # 2. Estimate crowd in dense regions
    dense_count_total = 0.0
    h_img, w_img, _ = image.shape
    
    for r_box in dense_crop_regions:
        rx1, ry1, rx2, ry2 = r_box
        # Clip coordinates to image boundary
        x1 = max(0, int(round(rx1)))
        y1 = max(0, int(round(ry1)))
        x2 = min(w_img, int(round(rx2)))
        y2 = min(h_img, int(round(ry2)))
        
        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
            
            # Call density model
            if hasattr(density_model, 'predict'):
                density_map = density_model.predict(crop)
            elif callable(density_model):
                density_map = density_model(crop)
            elif hasattr(density_model, 'forward'):
                density_map = density_model.forward(crop)
            else:
                raise ValueError("Provided density_model is not callable or has no recognizable predict/forward method.")
                
            # If the model returns a scalar count directly instead of a 2D map, handle it
            if isinstance(density_map, (int, float)):
                dense_count_total += float(density_map)
            elif isinstance(density_map, np.ndarray):
                dense_count_total += float(np.sum(density_map))
            else:
                dense_count_total += float(density_map)
                
    return float(filtered_sparse_count) + dense_count_total
