from typing import List, Dict, Any

def parse_yolo_results(results: List[Any]) -> List[Dict[str, Any]]:
    """
    Parses the bounding box metadata from the first Results object in a list of YOLO predictions.

    This function extracts the class ID, confidence score, and bounding box coordinates in both
    xyxy (absolute corners) and xywh (absolute center and size) formats for each detected object
    in the first result of the YOLO model's predictions.

    Args:
        results (List[Any]): A list of Ultralytics YOLO Results objects returned by a prediction.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing detection details:
            - "class_id" (int): The integer index representing the detected object's class.
            - "confidence" (float): The detection confidence score between 0.0 and 1.0.
            - "bbox_xyxy" (List[float]): Bounding box coordinates [xmin, ymin, xmax, ymax] in pixels.
            - "bbox_xywh" (List[float]): Bounding box coordinates [x_center, y_center, width, height] in pixels.
    """
    if not results:
        return []
        
    # Get the first Results object
    first_result = results[0]
    
    # Retrieve the Boxes object
    boxes = first_result.boxes
    if boxes is None or len(boxes) == 0:
        return []
        
    parsed_detections = []
    
    # Iterate through the detected bounding boxes
    for i in range(len(boxes)):
        # Extract class ID and confidence score as native Python scalar types
        class_id = int(boxes.cls[i].cpu().item())
        confidence = float(boxes.conf[i].cpu().item())
        
        # Convert bounding box tensors to list of floats
        bbox_xyxy = [float(val) for val in boxes.xyxy[i].cpu().numpy()]
        bbox_xywh = [float(val) for val in boxes.xywh[i].cpu().numpy()]
        
        parsed_detections.append({
            "class_id": class_id,
            "confidence": confidence,
            "bbox_xyxy": bbox_xyxy,
            "bbox_xywh": bbox_xywh
        })
        
    return parsed_detections
