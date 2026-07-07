from pathlib import Path
from typing import Union
from ultralytics import YOLO

def export_yolo_model(
    model_path: Union[str, Path],
    format_type: str,
    half_precision: bool = False,
    dynamic_shapes: bool = False
) -> str:
    """
    Load a YOLO model and export it to an optimized serialization format.

    Args:
        model_path (Union[str, Path]): Path to the PyTorch YOLO model file (e.g., 'yolo11n.pt').
        format_type (str): The target format for export (e.g., 'onnx', 'engine', 'tflite', 'openvino').
        half_precision (bool): If True, export the model with FP16 (half precision), which halves
                               the file size and speeds up inference on compatible GPU hardware.
                               Defaults to False.
        dynamic_shapes (bool): If True, allow dynamic input batch size and dimensions.
                               Defaults to False.

    Returns:
        str: The file path (or directory path) to the exported model.
    """
    model = YOLO(model_path)
    exported_path = model.export(
        format=format_type,
        half=half_precision,
        dynamic=dynamic_shapes
    )
    return str(exported_path)
