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
    x_min, y_min, x_max, y_max = rgb_fire_box
    
    # 1. Define all four corners of the RGB bounding box in homogeneous coordinates [x, y, 1]^T
    corners = np.array([
        [x_min, y_min, 1.0],  # Top-Left
        [x_max, y_min, 1.0],  # Top-Right
        [x_min, y_max, 1.0],  # Bottom-Left
        [x_max, y_max, 1.0]   # Bottom-Right
    ]).T  # Shape: (3, 4)
    
    # 2. Project corners using the 3x3 homography matrix
    projected = np.dot(homography_matrix, corners)  # Shape: (3, 4)
    
    # 3. Perform homogeneous division/normalization
    # Avoid division by zero by adding a small epsilon or filtering zero scale factors
    scale = projected[2]
    # In practice, scale should be positive and non-zero
    scale = np.where(np.abs(scale) < 1e-6, 1e-6, scale)
    
    proj_x = projected[0] / scale
    proj_y = projected[1] / scale
    
    # 4. Find the enclosing bounding box on the thermal grid
    tx_min = int(np.floor(np.min(proj_x)))
    tx_max = int(np.ceil(np.max(proj_x)))
    ty_min = int(np.floor(np.min(proj_y)))
    ty_max = int(np.ceil(np.max(proj_y)))
    
    # 5. Clamp coordinate limits to the boundaries of the thermal image
    h_t, w_t = thermal_image.shape
    tx_min = max(0, min(w_t - 1, tx_min))
    tx_max = max(0, min(w_t, tx_max))
    ty_min = max(0, min(h_t - 1, ty_min))
    ty_max = max(0, min(h_t, ty_max))
    
    # 6. Crop and verify
    if tx_min >= tx_max or ty_min >= ty_max:
        return False, 0.0, 0.0
        
    thermal_crop = thermal_image[ty_min:ty_max, tx_min:tx_max]
    
    if thermal_crop.size == 0:
        return False, 0.0, 0.0
        
    max_temp = float(np.max(thermal_crop))
    mean_temp = float(np.mean(thermal_crop))
    
    is_verified = max_temp >= temp_threshold
    
    return is_verified, max_temp, mean_temp
