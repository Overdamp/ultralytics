from pathlib import Path
from typing import Union

def visualize_and_save(
    model_path: Union[str, Path], 
    image_path: Union[str, Path], 
    output_path: Union[str, Path]
) -> None:
    """
    Loads a YOLO model, performs inference on a single image, 
    renders the prediction bounding boxes and labels using results[0].plot(), 
    and saves the annotated image to output_path.

    Args:
        model_path (Union[str, Path]): Path to the YOLO model weight file (.pt).
        image_path (Union[str, Path]): Path to the input image file.
        output_path (Union[str, Path]): Path where the annotated output image will be saved.
    """
    # TODO: Load the YOLO model using the YOLO class
    # TODO: Perform inference on the image using the model
    # TODO: Extract the first Result object from the results list
    # TODO: Render prediction bounding boxes and labels onto the image using results[0].plot()
    # TODO: Ensure the directory path of output_path exists
    # TODO: Save the annotated image (returned by results[0].plot() as BGR NumPy array) using cv2/PIL
    pass
