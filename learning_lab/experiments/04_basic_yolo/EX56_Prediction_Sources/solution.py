from typing import List, Any
from ultralytics import YOLO
from ultralytics.engine.results import Results

def run_inference(model_path: str, source: Any, save_results: bool = True) -> List[Results]:
    """
    Runs model inference on a specified source using the Ultralytics YOLO framework.

    This function loads a YOLO model checkpoint and performs object detection prediction
    on the provided source. The source can be a path to an image/video, a directory,
    a camera stream index (e.g., 0), a URL, a NumPy array, or a PIL Image.

    Args:
        model_path (str): Path to the YOLO model file (e.g., 'yolo11n.pt').
        source (Any): The input source for inference. Can be a string path, integer webcam index,
                      NumPy array (OpenCV format), PIL Image, or URL.
        save_results (bool, optional): Whether to save the annotated inference results to disk. 
                                       Defaults to True.

    Returns:
        List[Results]: A list of Ultralytics Results objects containing the prediction outputs.
    """
    # Load the YOLO model from the specified path
    model = YOLO(model_path)
    
    # Run inference on the input source using the model's .predict() method
    results = model.predict(source=source, save=save_results)
    
    # Return the list of Results objects
    return results
