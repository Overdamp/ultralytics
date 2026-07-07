import subprocess
from typing import Optional

def run_yolo_cli_command(
    task: Optional[str],
    mode: str,
    model: str,
    data: Optional[str] = None,
    epochs: Optional[int] = None,
    imgsz: Optional[int] = None
) -> subprocess.CompletedProcess:
    """
    Constructs an Ultralytics YOLO CLI command string and executes it using subprocess.run.

    Args:
        task (Optional[str]): The computer vision task (e.g., 'detect', 'segment', 'classify', 'pose', 'obb').
                              Can be None if the task is inferred from the model.
        mode (str): The operational mode (e.g., 'train', 'val', 'predict', 'export', 'track', 'benchmark').
        model (str): Path to model file or model name (e.g., 'yolov8n.pt', 'yolov8n.yaml').
        data (Optional[str]): Path to the data configuration file (e.g., 'coco8.yaml').
        epochs (Optional[int]): Number of epochs for training.
        imgsz (Optional[int]): Image resolution size.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess execution.
    """
    cmd = ["yolo"]
    if task:
        cmd.append(task)
    cmd.append(mode)
    cmd.append(f"model={model}")
    
    if data is not None:
        cmd.append(f"data={data}")
    if epochs is not None:
        cmd.append(f"epochs={epochs}")
    if imgsz is not None:
        cmd.append(f"imgsz={imgsz}")
        
    print(f"Executing CLI command: {' '.join(cmd)}")
    
    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result
