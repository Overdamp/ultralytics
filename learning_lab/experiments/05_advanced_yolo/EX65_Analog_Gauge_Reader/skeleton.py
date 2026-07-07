import math
from typing import Tuple

def get_needle_reading(
    center_kpt: Tuple[float, float],
    tip_kpt: Tuple[float, float],
    min_val: float,
    max_val: float,
    min_angle: float,
    max_angle: float
) -> float:
    """
    Calculates the reading of an analog gauge needle using keypoints.
    
    This function computes the needle's angle relative to the horizontal axis (3 o'clock)
    using trigonometry (math.atan2) in image space (where the y-axis points downwards),
    normalizes the angle, clamps it within the calibrated limits, and performs
    linear interpolation to map the angle to the final physical gauge reading.
    
    Mathematical details:
    1. Calculate dx = tip_x - center_x
    2. Calculate dy = -(tip_y - center_y)  [Invert dy since y increases downwards in image space]
    3. Calculate theta = arctan2(dy, dx) in degrees
    4. Normalize theta to [-180, 180] range
    5. Clamp theta within [min_angle, max_angle]
    6. Interpolate: val = min_val + ((theta - min_angle) / (max_angle - min_angle)) * (max_val - min_val)
    
    Args:
        center_kpt (Tuple[float, float]): Coordinates (x, y) of the needle pivot center.
        tip_kpt (Tuple[float, float]): Coordinates (x, y) of the needle tip.
        min_val (float): Physical value of the gauge at the minimum angle (e.g., 0.0 psi).
        max_val (float): Physical value of the gauge at the maximum angle (e.g., 100.0 psi).
        min_angle (float): The angle (in degrees) corresponding to min_val.
        max_angle (float): The angle (in degrees) corresponding to max_val.
        
    Returns:
        float: The calibrated physical reading of the gauge.
    """
    # TODO: Calculate horizontal and vertical differences (dx and dy)
    # Hint: Remember that image coordinates have y pointing down, so dy must be inverted
    
    # TODO: Compute angle in degrees using math.atan2 and math.degrees
    
    # TODO: Normalize the angle to be within [-180, 180] if necessary
    
    # TODO: Clamp the calculated angle to the min/max angle boundaries
    
    # TODO: Perform linear interpolation to map the clamped angle to the physical value range
    
    # TODO: Return the computed reading
    pass
