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
    # TODO: Calculate the intersection ratio of each sparse box with all dense regions
    # Hint: Keep a sparse box only if its maximum overlap ratio with any dense region is below overlap_threshold
    
    # TODO: Calculate the sparse crowd count (number of remaining sparse boxes)
    
    # TODO: Calculate the dense crowd count
    # For each region in dense_crop_regions:
    #   1. Crop the region from the image.
    #   2. Feed the crop to the density_model to get the density map.
    #   3. Sum all pixel values of the density map (integration) to get the count for that region.
    #   4. Add to the dense count total.
    
    # TODO: Return the sum of the sparse and dense counts
    pass
