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
    # TODO: Check if results list is empty; if so, return an empty list
    
    # TODO: Get the first Results object (results[0])
    
    # TODO: Retrieve the Boxes object from the first result
    
    # TODO: Iterate through the boxes and extract class_id, confidence, bbox_xyxy, and bbox_xywh.
    # Make sure to convert PyTorch tensors to standard Python types (e.g. CPU memory, float, int, list of floats)
    
    # TODO: Compile and return the list of parsed dictionaries
    pass
