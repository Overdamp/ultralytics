from ultralytics import YOLO
from typing import Any

def resume_yolo_training(checkpoint_path: str) -> Any:
    """
    Loads an interrupted training checkpoint (usually last.pt) and resumes training.

    Args:
        checkpoint_path (str): Path to the model checkpoint file (e.g., 'runs/detect/train/weights/last.pt').

    Returns:
        Any: The training results object.
    """
    # TODO: Load the checkpointed model from checkpoint_path
    # TODO: Call train with resume=True to continue training
    # TODO: Return the training results
    pass
