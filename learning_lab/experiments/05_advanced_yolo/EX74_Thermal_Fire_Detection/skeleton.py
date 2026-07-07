import numpy as np
from typing import List, Tuple

def verify_fire_anomaly(
    rgb_fire_box: List[float],
    thermal_image: np.ndarray,
    homography_matrix: np.ndarray,
    temp_threshold: float = 80.0
) -> Tuple[bool, float, float]:
    """
    Verifies if a visual fire detection candidate from an RGB camera corresponds 
    to a genuine high-heat signature on a co-registered thermal camera.

    RGB-based fire detection is highly prone to false positives (e.g. orange vests, 
    emergency lights). To reduce false alarms, this function uses a homography matrix 
    (obtained via dual-camera calibration) to project the RGB bounding box coordinates 
    onto the Long-Wave Infrared (LWIR) thermal grid. It then extracts temperature 
    statistics from the corresponding projected area to verify the heat signature.

    Args:
        rgb_fire_box (List[float]): Bounding box from the RGB camera in pixel coordinates:
                                    [x_min, y_min, x_max, y_max].
        thermal_image (np.ndarray): Co-registered thermal image (shape HxW) containing 
                                    calibrated temperature values in Celsius.
        homography_matrix (np.ndarray): A 3x3 homography matrix that maps coordinates from 
                                        the RGB image plane to the thermal image plane.
        temp_threshold (float): The safety temperature threshold in Celsius (e.g. 80.0°C). 
                                If the maximum temperature in the projected area meets or 
                                exceeds this threshold, the fire is verified.

    Returns:
        Tuple[bool, float, float]: A tuple containing:
                                   - is_verified (bool): True if the heat signature is verified, 
                                                         False otherwise.
                                   - max_temp (float): The maximum temperature inside the projected region.
                                   - mean_temp (float): The mean temperature inside the projected region.
    """
    # TODO: Extract the corner coordinates of the RGB bounding box.
    # To handle potential perspective distortions (skew, rotation) introduced by homography projection,
    # define all four corners of the bounding box:
    # 1. Top-Left: [x_min, y_min]
    # 2. Top-Right: [x_max, y_min]
    # 3. Bottom-Left: [x_min, y_max]
    # 4. Bottom-Right: [x_max, y_max]
    
    # TODO: Project all four corners from the RGB coordinate space to the thermal coordinate space
    # using homogeneous matrix multiplication: x_thermal ~ H * x_rgb.
    # Be sure to perform homogeneous normalization (divide x and y by the third component) for each point.
    
    # TODO: Find the bounding box on the thermal grid that encloses all four projected corners:
    # tx_min = min(projected_x_coordinates)
    # tx_max = max(projected_x_coordinates)
    # ty_min = min(projected_y_coordinates)
    # ty_max = max(projected_y_coordinates)
    
    # TODO: Crop the region defined by [ty_min:ty_max, tx_min:tx_max] from the thermal image.
    # Make sure to clamp these coordinates to the boundary of the thermal image to prevent out-of-bound errors.
    
    # TODO: If the crop is empty, return (False, 0.0, 0.0).
    
    # TODO: Compute the maximum and mean temperatures of the cropped thermal region.
    
    # TODO: Compare the maximum temperature to temp_threshold to verify the fire anomaly.
    
    pass
