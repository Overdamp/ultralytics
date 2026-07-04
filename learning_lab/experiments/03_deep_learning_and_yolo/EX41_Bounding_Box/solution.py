import numpy as np

def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from [xc, yc, w, h] to [x1, y1, x2, y2].
    Can handle single box list or a NumPy array of shape (N, 4).
    """
    boxes = np.array(boxes)
    converted = np.zeros_like(boxes, dtype=float)
    converted[..., 0] = boxes[..., 0] - (boxes[..., 2] / 2.0)
    converted[..., 1] = boxes[..., 1] - (boxes[..., 3] / 2.0)
    converted[..., 2] = boxes[..., 0] + (boxes[..., 2] / 2.0)
    converted[..., 3] = boxes[..., 1] + (boxes[..., 3] / 2.0)
    return converted

def xyxy_to_xywh(boxes):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [xc, yc, w, h].
    """
    boxes = np.array(boxes)
    converted = np.zeros_like(boxes, dtype=float)
    converted[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2.0
    converted[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2.0
    converted[..., 2] = boxes[..., 2] - boxes[..., 0]
    converted[..., 3] = boxes[..., 3] - boxes[..., 1]
    return converted

if __name__ == "__main__":
    box_xyxy = [100.0, 150.0, 300.0, 400.0]
    box_xywh = xyxy_to_xywh(box_xyxy)
    box_restored = xywh_to_xyxy(box_xywh)
    
    print("--- Training Results ---")
    print(f"Original XYXY  : [{box_xyxy[0]:.1f}, {box_xyxy[1]:.1f}, {box_xyxy[2]:.1f}, {box_xyxy[3]:.1f}]")
    print(f"Converted XYWH : [{box_xywh[0]:.1f}, {box_xywh[1]:.1f}, {box_xywh[2]:.1f}, {box_xywh[3]:.1f}]")
    print(f"Restored XYXY  : [{box_restored[0]:.1f}, {box_restored[1]:.1f}, {box_restored[2]:.1f}, {box_restored[3]:.1f}]")
