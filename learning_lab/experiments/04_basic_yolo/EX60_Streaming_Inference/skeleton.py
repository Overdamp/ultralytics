from typing import Generator, List, Dict, Any
from ultralytics import YOLO

def stream_inference_generator(
    model_path: str,
    video_path: str
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Perform memory-safe streaming inference on a video source, parsing and yielding
    bounding box detections frame-by-frame.

    This generator utilizes `stream=True` to process video frames dynamically, keeping
    memory utilization to a minimum (O(1) space complexity relative to video length).

    Args:
        model_path (str): Path to the pre-trained YOLO model (e.g., 'yolo11n.pt').
        video_path (str): Path to the input video file or stream source.

    Yields:
        List[Dict[str, Any]]: A list of dictionaries representing the detected bounding boxes
                               for the current frame. Each dictionary contains:
                               - "box": list of float [x_min, y_min, x_max, y_max]
                               - "class_id": int
                               - "class_name": str
                               - "confidence": float
    """
    # TODO: Load the YOLO model
    # TODO: Perform inference using model.predict() with stream=True
    # TODO: Iterate through the generator, parsing coordinates, class IDs, names, and confidences
    # TODO: Yield the list of parsed dictionaries frame-by-frame
    pass
