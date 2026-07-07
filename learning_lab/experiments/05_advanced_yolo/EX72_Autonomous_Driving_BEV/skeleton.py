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
    # TODO: Calculate the bottom-center pixel coordinate (u, v) of the 2D bounding box.
    # The x-coordinate (u) is the midpoint of the horizontal range, and the y-coordinate (v) 
    # is the bottom edge of the box.
    
    # TODO: Form a homogeneous coordinate vector [u, v, 1]^T.
    
    # TODO: Perform matrix multiplication between the homography matrix and the homogeneous pixel vector
    # to obtain the ground homogeneous coordinates.
    
    # TODO: Perform homogeneous normalization (divide the projected X and Y coordinates by the scaling factor/Z coordinate).
    # Ensure to check for division by zero.
    
    # TODO: Return the projected (X_w, Y_w) ground coordinates as float values.
    pass
