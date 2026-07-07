import numpy as np
from typing import Tuple

def grade_fruit_contour(
    mask: np.ndarray,
    image_hsl: np.ndarray
) -> Tuple[int, float, str]:
    """
    Evaluates the quality of a piece of produce by calculating its segmented pixel area (size)
    and analyzing the average Hue channel value within the segment to determine ripeness.

    In smart agriculture sorting systems, bounding boxes are insufficient because they include 
    background pixels. Using YOLO Instance Segmentation (YOLO-seg), we obtain precise binary 
    masks. Decoupled color spaces like HSL or HSV are utilized because the Hue channel 
    is invariant to shadows and uneven illumination, allowing for robust ripeness classification.

    Args:
        mask (np.ndarray): Binary segmentation mask (shape HxW, dtype bool or uint8).
                           A value of True/1 indicates the pixel belongs to the fruit.
        image_hsl (np.ndarray): HSL (or HSV) representation of the image (shape HxWxC).
                                The first channel (channel 0) represents the Hue.

    Returns:
        Tuple[int, float, str]: A tuple containing:
                                - pixel_area (int): The total count of pixels belonging to the fruit.
                                - mean_hue (float): The average Hue angle (in degrees, range [0.0, 360.0]) 
                                                    of the fruit's pixels.
                                - ripeness_grade (str): The ripeness classification:
                                                       - "Unripe (Green)" (60° <= Mean Hue <= 150°)
                                                       - "Ripe (Yellow/Orange)" (20° <= Mean Hue < 60°)
                                                       - "Overripe (Red)" (Mean Hue < 20° or Mean Hue > 150°)
    """
    # 1. Calculate the total pixel area of the mask (count of True/1 pixels)
    # Convert mask to boolean array for robust indexing
    mask_bool = mask.astype(bool)
    pixel_area = int(np.sum(mask_bool))
    
    if pixel_area == 0:
        return 0, 0.0, "Unknown"
        
    # 2. Extract the Hue values of the pixels that fall inside the mask
    hue_channel = image_hsl[:, :, 0]
    hue_inside_mask = hue_channel[mask_bool]
    
    # 3. Scale the Hue channel to degrees [0.0, 360.0]
    # Check the global max of the Hue channel to determine scaling
    global_max_hue = np.max(hue_channel)
    
    if global_max_hue <= 1.0:
        # Normalized float representation [0, 1]
        hue_inside_mask_deg = hue_inside_mask * 360.0
    elif global_max_hue <= 180.0 and image_hsl.dtype == np.uint8:
        # Standard OpenCV uint8 HSV/HSL representation where H is [0, 180]
        hue_inside_mask_deg = hue_inside_mask.astype(np.float32) * 2.0
    else:
        # Already scaled in degrees [0, 360]
        hue_inside_mask_deg = hue_inside_mask.astype(np.float32)
        
    # 4. Calculate the mean of these Hue degrees
    mean_hue = float(np.mean(hue_inside_mask_deg))
    
    # 5. Categorize the fruit ripeness based on the average Hue angle
    if 60.0 <= mean_hue <= 150.0:
        ripeness_grade = "Unripe (Green)"
    elif 20.0 <= mean_hue < 60.0:
        ripeness_grade = "Ripe (Yellow/Orange)"
    else:
        ripeness_grade = "Overripe (Red)"
        
    return pixel_area, mean_hue, ripeness_grade
