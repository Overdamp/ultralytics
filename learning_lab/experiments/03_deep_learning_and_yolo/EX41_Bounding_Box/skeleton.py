import numpy as np

def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from [xc, yc, w, h] to [x1, y1, x2, y2].
    Can handle single box list or a NumPy array of shape (N, 4).
    """
    boxes = np.array(boxes)
    converted = np.zeros_like(boxes, dtype=float)
    # TODO: Implement conversion from [xc, yc, w, h] to [x1, y1, x2, y2]
    # Hint: x1 = xc - w / 2, y1 = yc - h / 2, x2 = xc + w / 2, y2 = yc + h / 2
    
    return converted

def xyxy_to_xywh(boxes):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [xc, yc, w, h].
    """
    boxes = np.array(boxes)
    converted = np.zeros_like(boxes, dtype=float)
    # TODO: Implement conversion from [x1, y1, x2, y2] to [xc, yc, w, h]
    # Hint: xc = (x1 + x2) / 2, yc = (y1 + y2) / 2, w = x2 - x1, h = y2 - y1
    
    return converted

if __name__ == "__main__":
    box_xyxy = [100.0, 150.0, 300.0, 400.0]
    box_xywh = xyxy_to_xywh(box_xyxy)
    box_restored = xywh_to_xyxy(box_xywh)
    
    print("--- Training Results ---")
    if np.any(box_xywh != 0.0):
        print(f"Original XYXY  : [{box_xyxy[0]:.1f}, {box_xyxy[1]:.1f}, {box_xyxy[2]:.1f}, {box_xyxy[3]:.1f}]")
        print(f"Converted XYWH : [{box_xywh[0]:.1f}, {box_xywh[1]:.1f}, {box_xywh[2]:.1f}, {box_xywh[3]:.1f}]")
        print(f"Restored XYXY  : [{box_restored[0]:.1f}, {box_restored[1]:.1f}, {box_restored[2]:.1f}, {box_restored[3]:.1f}]")
    else:
        print("Bounding box coordinate conversions not implemented yet.")
