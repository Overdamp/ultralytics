import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

def audit_ppe_compliance(
    person_crops: List[np.ndarray],
    ppe_detector: YOLO,
    conf_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Audits PPE compliance by running a secondary YOLO model on cropped images of people.
    
    This function implements a hierarchical inspection pattern:
    1. For each crop of a person, it defines vertical regions of interest (ROIs):
       - Head Zone (top 25% of the person's height)
       - Torso Zone (middle 20% to 75% of the person's height)
    2. Slices the crop into head and torso sub-crops.
    3. Runs the secondary `ppe_detector` on each zone sub-crop.
    4. Evaluates if the person is wearing a helmet (detected in the Head Zone)
       and/or a safety vest (detected in the Torso Zone) by looking up class names
       (or fallbacks if names are not set).
    5. Returns safety audit results for each person crop.
    
    Args:
        person_crops (List[np.ndarray]): A list of image crops of individual people.
        ppe_detector (YOLO): A YOLO model trained to detect PPE objects (e.g., helmet, vest).
        conf_threshold (float): Confidence threshold for detections.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, one for each person crop, with:
            - 'helmet_worn': bool (True if helmet detected in the head crop)
            - 'vest_worn': bool (True if safety vest detected in the torso crop)
    """
    # TODO: Identify class indices for 'helmet' (or 'hard-hat') and 'vest' from ppe_detector.names
    # Hint: Loop through ppe_detector.names to find matching classes dynamically
    
    # TODO: Process each person crop
    # For each crop:
    #   1. Check if the crop is empty (size is 0)
    #   2. Compute coordinates to slice Head Zone (top 25%) and Torso Zone (middle 20% to 75%)
    #   3. Feed the head crop into the ppe_detector and check for a helmet detection
    #   4. Feed the torso crop into the ppe_detector and check for a vest detection
    #   5. Append the boolean flags to the results list
    
    # TODO: Return the audit list
    pass
