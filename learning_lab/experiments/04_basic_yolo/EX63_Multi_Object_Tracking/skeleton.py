from pathlib import Path
from typing import List, Union

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
    # TODO: Load the YOLO model using the YOLO class
    # TODO: Run tracking on the video source with stream=True, persist=True, and the selected tracker
    # TODO: Iterate through the frames in the results stream
    # TODO: In each frame, check if results[0].boxes and results[0].boxes.id are not None
    # TODO: Extract the tracking IDs, convert to integers, and add to a set of unique IDs
    # TODO: Return the sorted list of unique active tracking IDs
    pass
