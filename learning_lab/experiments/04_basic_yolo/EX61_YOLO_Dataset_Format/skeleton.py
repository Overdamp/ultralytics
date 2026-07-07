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
    # TODO: Extract coordinates from bbox_pixel
    # TODO: Calculate bounding box width and height in pixels
    # TODO: Calculate bounding box center (x, y) in pixels
    # TODO: Normalize center coordinates and dimensions by image dimensions
    # TODO: Return formatted YOLO coordinate string with 6 decimal places
    pass
