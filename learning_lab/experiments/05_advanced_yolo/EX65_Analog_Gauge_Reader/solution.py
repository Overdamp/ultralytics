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
    # Calculate horizontal and vertical differences (dx and dy)
    dx = tip_kpt[0] - center_kpt[0]
    dy = -(tip_kpt[1] - center_kpt[1])  # Invert dy since y increases downwards in image space
    
    # Compute angle in degrees using math.atan2 and math.degrees
    angle = math.degrees(math.atan2(dy, dx))
    
    # Normalize the angle to be within [-180, 180] range
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
        
    # Clamp the calculated angle to the min/max angle boundaries
    clamped_angle = max(min(angle, max_angle), min_angle)
    
    # Perform linear interpolation to map the clamped angle to the physical value range
    angle_span = max_angle - min_angle
    if angle_span == 0:
        return min_val
        
    fraction = (clamped_angle - min_angle) / angle_span
    reading = min_val + fraction * (max_val - min_val)
    
    return reading
