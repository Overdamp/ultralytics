from pathlib import Path
from typing import Union
import cv2
from ultralytics import YOLO

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
    # Load the YOLO model
    model = YOLO(str(model_path))
    
    # Run inference on the image (save=False as we process the visualization ourselves)
    results = model(str(image_path))
    
    # Extract the first prediction result (single image input)
    result = results[0]
    
    # Render prediction bounding boxes and labels
    # plot() returns a BGR numpy array compatible with OpenCV
    annotated_frame = result.plot()
    
    # Ensure the destination directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the annotated frame to output_path using OpenCV
    cv2.imwrite(str(output_path), annotated_frame)
