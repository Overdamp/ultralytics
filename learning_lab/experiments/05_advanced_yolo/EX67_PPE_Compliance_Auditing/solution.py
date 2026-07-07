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
    audit_report = []
    
    # Identify class indices for 'helmet' and 'vest' dynamically
    helmet_ids = []
    vest_ids = []
    
    if hasattr(ppe_detector, 'names') and ppe_detector.names:
        for idx, name in ppe_detector.names.items():
            name_lower = name.lower()
            if any(k in name_lower for k in ['helmet', 'hat', 'hard-hat', 'head']):
                helmet_ids.append(int(idx))
            if 'vest' in name_lower:
                vest_ids.append(int(idx))
                
    # Fallback to defaults (0 = helmet/hard-hat, 1 = safety-vest/vest) if classes are not found
    if not helmet_ids:
        helmet_ids = [0]
    if not vest_ids:
        vest_ids = [1]
        
    for crop in person_crops:
        if crop is None or crop.size == 0:
            audit_report.append({"helmet_worn": False, "vest_worn": False})
            continue
            
        h, w, _ = crop.shape
        
        # 1. Slice Head Zone: top 25%
        head_crop = crop[0:int(h * 0.25), 0:w]
        
        # 2. Slice Torso Zone: middle 20% to 75%
        torso_crop = crop[int(h * 0.20):int(h * 0.75), 0:w]
        
        # Initialize flags
        has_helmet = False
        has_vest = False
        
        # 3. Detect helmet in Head Zone
        if head_crop.size > 0:
            head_results = ppe_detector(head_crop, conf=conf_threshold, verbose=False)[0]
            if len(head_results.boxes) > 0:
                has_helmet = any(
                    int(box.cls[0]) in helmet_ids and float(box.conf[0]) >= conf_threshold
                    for box in head_results.boxes
                )
                
        # 4. Detect vest in Torso Zone
        if torso_crop.size > 0:
            torso_results = ppe_detector(torso_crop, conf=conf_threshold, verbose=False)[0]
            if len(torso_results.boxes) > 0:
                has_vest = any(
                    int(box.cls[0]) in vest_ids and float(box.conf[0]) >= conf_threshold
                    for box in torso_results.boxes
                )
                
        audit_report.append({
            "helmet_worn": has_helmet,
            "vest_worn": has_vest
        })
        
    return audit_report
