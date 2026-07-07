from ultralytics import YOLO

def get_yolo_model_by_task(task_type: str, size: str = "n") -> YOLO:
    """
    Map a computer vision task type to the corresponding pre-trained YOLO11 model filename
    and instantiate the model.

    Supported task types:
        - "detect": Standard Object Detection (e.g., 'yolo11n.pt')
        - "segment": Instance Segmentation (e.g., 'yolo11n-seg.pt')
        - "classify": Image Classification (e.g., 'yolo11n-cls.pt')
        - "pose": Keypoint / Pose Estimation (e.g., 'yolo11n-pose.pt')
        - "obb": Oriented Bounding Box Detection (e.g., 'yolo11n-obb.pt')

    Args:
        task_type (str): The desired task. One of: "detect", "segment", "classify", "pose", "obb".
        size (str): Model size/scale identifier. Usually one of "n" (nano), "s" (small), "m" (medium),
                    "l" (large), "x" (xlarge). Defaults to "n".

    Returns:
        YOLO: An instantiated YOLO model object configured for the specified task.

    Raises:
        ValueError: If the task_type is not one of the five supported tasks.
    """
    task_mapping = {
        "detect": "",
        "segment": "-seg",
        "classify": "-cls",
        "pose": "-pose",
        "obb": "-obb"
    }

    normalized_task = task_type.lower().strip()
    if normalized_task not in task_mapping:
        raise ValueError(
            f"Unsupported task_type '{task_type}'. "
            f"Supported tasks are: {list(task_mapping.keys())}"
        )

    suffix = task_mapping[normalized_task]
    model_name = f"yolo11{size}{suffix}.pt"
    
    return YOLO(model_name)
