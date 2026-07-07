from typing import List, Union

def convert_to_yolo_format(
    bbox_pixel: List[Union[int, float]], 
    img_w: int, 
    img_h: int, 
    class_id: int
) -> str:
    """
    Converts raw pixel bounding box coordinates [xmin, ymin, xmax, ymax]
    into the normalized YOLO format string: "<class_id> <x_center> <y_center> <width> <height>".

    Mathematical formulation:
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

    Args:
        bbox_pixel (List[Union[int, float]]): Bounding box in pixels as [xmin, ymin, xmax, ymax].
        img_w (int): The width of the image in pixels.
        img_h (int): The height of the image in pixels.
        class_id (int): The integer class label for the object.

    Returns:
        str: A string formatted as "<class_id> <x_center> <y_center> <width> <height>" 
             where float values are formatted to 6 decimal places.
    """
    xmin, ymin, xmax, ymax = bbox_pixel
    
    # Calculate box dimensions
    box_w = xmax - xmin
    box_h = ymax - ymin
    
    # Calculate center coordinates
    x_center = xmin + (box_w / 2.0)
    y_center = ymin + (box_h / 2.0)
    
    # Normalize coordinates
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    width_norm = box_w / img_w
    height_norm = box_h / img_h
    
    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
