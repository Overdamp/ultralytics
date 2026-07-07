from ultralytics import YOLO
from typing import Union, Any

def train_custom_model(
    model_path: str,
    data_yaml_path: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: Union[int, str]
) -> Any:
    """
    Loads a YOLO model from model_path and trains it using the specified parameters.

    Args:
        model_path (str): Path to the model file (e.g., 'yolov8n.pt' or 'yolov8n.yaml').
        data_yaml_path (str): Path to the dataset YAML configuration file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        imgsz (int): Resolution to resize input images.
        device (Union[int, str]): Execution device (e.g., 0, 'cpu', 'cuda').

    Returns:
        Any: The training results object returned by the model.train() method.
    """
    model = YOLO(model_path)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device
    )
    return results
