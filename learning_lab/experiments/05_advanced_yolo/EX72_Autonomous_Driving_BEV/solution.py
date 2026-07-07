import numpy as np
from typing import List, Tuple

def project_to_bev(
    bbox_2d: List[float],
    homography_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Projects the bottom-center point of a 2D bounding box in perspective space 
    onto a 2D ground plane (Bird's Eye View coordinate system) using Inverse Perspective Mapping (IPM).

    In autonomous driving, path planning and tracking algorithms operate in a 2D metric 
    ground grid (BEV) rather than 2D pixel space. Under the flat-ground assumption (Z_w = 0),
    we can project image plane coordinates back to the road plane using a homography matrix.
    We project the bottom-center of the bounding box because that is where the vehicle 
    touches the road surface.

    Args:
        bbox_2d (List[float]): Bounding box in pixel coordinates: [x_min, y_min, x_max, y_max].
        homography_matrix (np.ndarray): A 3x3 homography matrix that maps coordinates from 
                                        the image plane to the BEV ground plane (image-to-ground homography).

    Returns:
        Tuple[float, float]: Coordinates of the projected point on the ground plane (X_w, Y_w) in meters.
                             - X_w: Lateral distance relative to the camera base (positive is right, negative is left).
                             - Y_w: Longitudinal distance (distance ahead on the road).
    """
    # 1. Calculate the bottom-center pixel coordinate (u, v) of the 2D bounding box
    x_min, y_min, x_max, y_max = bbox_2d
    u = (x_min + x_max) / 2.0
    v = y_max  # The contact point with the road is the bottom of the bounding box
    
    # 2. Form a homogeneous coordinate vector [u, v, 1]^T
    pixel_coord = np.array([[u], [v], [1.0]], dtype=np.float32)
    
    # 3. Project to the ground plane by multiplying the homography matrix and the pixel coordinate
    ground_homogeneous = np.dot(homography_matrix, pixel_coord)
    
    # 4. Perform homogeneous normalization (divide by the third/scale component)
    scale = ground_homogeneous[2, 0]
    if abs(scale) > 1e-6:
        x_bev = ground_homogeneous[0, 0] / scale
        y_bev = ground_homogeneous[1, 0] / scale
    else:
        x_bev = 0.0
        y_bev = 0.0
        
    return float(x_bev), float(y_bev)
