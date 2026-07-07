from pathlib import Path
from typing import List, Union
from ultralytics import YOLO

def track_objects_in_video(
    model_path: Union[str, Path], 
    video_path: Union[str, Path], 
    tracker_config: str = "bytetrack.yaml"
) -> List[int]:
    """
    Performs multi-object tracking on a video source using YOLO's track method.
    Iterates through the video frame-by-frame (using stream=True to prevent memory overflow)
    and collects all unique active tracking IDs detected throughout the video.

    Args:
        model_path (Union[str, Path]): Path to the YOLO model weight file (.pt).
        video_path (Union[str, Path]): Path to the input video file.
        tracker_config (str): Tracker configuration file, e.g., 'bytetrack.yaml' or 'botsort.yaml'.

    Returns:
        List[int]: A sorted list of unique active tracking IDs detected in the video.
    """
    # Load the YOLO model
    model = YOLO(str(model_path))
    
    unique_ids = set()
    
    # Run tracking with stream=True to process frame-by-frame and free memory
    results = model.track(
        source=str(video_path), 
        tracker=tracker_config, 
        persist=True, 
        stream=True
    )
    
    for result in results:
        # Check if the result has boxes and if tracking IDs are available
        if result.boxes is not None and result.boxes.id is not None:
            # Convert tracking IDs tensor to a list of integers
            ids = result.boxes.id.int().cpu().tolist()
            unique_ids.update(ids)
            
    return sorted(list(unique_ids))
