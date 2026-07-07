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
    model = YOLO(model_path)
    class_names = model.names
    
    # Run inference in streaming mode
    results_generator = model.predict(source=video_path, stream=True)
    
    for result in results_generator:
        parsed_boxes = []
        if result.boxes is not None:
            boxes = result.boxes
            # Extract attributes to CPU using vectorized operations to release GPU references immediately
            clss = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxys = boxes.xyxy.cpu().tolist()
            
            for cls_val, conf_val, xyxy_val in zip(clss, confs, xyxys):
                cls_id = int(cls_val)
                parsed_boxes.append({
                    "box": xyxy_val,
                    "class_id": cls_id,
                    "class_name": class_names.get(cls_id, "unknown"),
                    "confidence": float(conf_val)
                })
        yield parsed_boxes
