import numpy as np

def decode_anchor_based(grid_x, grid_y, anchor_w, anchor_h, tx, ty, tw, th, stride=32):
    """
    Decodes bounding boxes from anchor-based predictions (e.g. YOLOv3/v5 style).
    
    Args:
        grid_x, grid_y (float): Grid cell coordinates (row, col indices).
        anchor_w, anchor_h (float): Base anchor box width and height in image space.
        tx, ty, tw, th (float): Network raw regression outputs.
        stride (int): Grid stride to map grid coordinates to image coordinates.
        
    Returns:
        np.ndarray: Decoded bounding box [x_min, y_min, x_max, y_max] in image coordinates.
    """
    # TODO: Calculate bounding box center (bx, by) in grid space using sigmoid adjustment
    # bx = sigmoid(tx) + grid_x
    # by = sigmoid(ty) + grid_y
    bx = 0.0
    by = 0.0
    
    # TODO: Calculate bounding box width and height (bw, bh) in image space
    # bw = anchor_w * exp(tw)
    # bh = anchor_h * exp(th)
    bw = 0.0
    bh = 0.0
    
    # TODO: Scale bx, by to image space using the stride
    bx_img = 0.0
    by_img = 0.0
    
    # TODO: Convert from [cx, cy, w, h] to [x_min, y_min, x_max, y_max] coordinates
    x_min = 0.0
    y_min = 0.0
    x_max = 0.0
    y_max = 0.0
    
    return np.array([x_min, y_min, x_max, y_max])

def decode_anchor_free(grid_x, grid_y, l, t, r, b, stride=32):
    """
    Decodes bounding boxes from anchor-free predictions (e.g. YOLOv8 style).
    
    Args:
        grid_x, grid_y (float): Grid cell coordinates (row, col indices).
        l, t, r, b (float): Network raw predictions: distance to left, top, right, bottom boundaries in grid units.
        stride (int): Grid stride to map grid coordinates to image coordinates.
        
    Returns:
        np.ndarray: Decoded bounding box [x_min, y_min, x_max, y_max] in image coordinates.
    """
    # TODO: Calculate the center point of the cell in image space
    # cx = (grid_x + 0.5) * stride
    # cy = (grid_y + 0.5) * stride
    cx = 0.0
    cy = 0.0
    
    # TODO: Calculate boundary coordinates in image space by applying left, top, right, bottom offsets
    x_min = 0.0
    y_min = 0.0
    x_max = 0.0
    y_max = 0.0
    
    return np.array([x_min, y_min, x_max, y_max])

if __name__ == "__main__":
    stride = 32
    grid_x, grid_y = 5.0, 5.0
    
    # 1. Anchor-based predictions
    anchor_w, anchor_h = 64.0, 128.0
    tx, ty, tw, th = 0.2, -0.4, 0.1, -0.2
    
    box_anchor_based = decode_anchor_based(grid_x, grid_y, anchor_w, anchor_h, tx, ty, tw, th, stride)
    
    # 2. Anchor-free predictions
    # Distance to boundaries (l, t, r, b) in grid units
    l, t, r, b = 1.2, 1.8, 1.5, 2.2
    
    box_anchor_free = decode_anchor_free(grid_x, grid_y, l, t, r, b, stride)
    
    print("--- Decoding Results ---")
    if not np.allclose(box_anchor_based, 0.0):
        print(f"Decoded Anchor-Based Box: {[round(float(coord), 2) for coord in box_anchor_based]}")
    else:
        print("Anchor-based decoding not implemented.")
        
    if not np.allclose(box_anchor_free, 0.0):
        print(f"Decoded Anchor-Free Box:  {[round(float(coord), 2) for coord in box_anchor_free]}")
    else:
        print("Anchor-free decoding not implemented.")
