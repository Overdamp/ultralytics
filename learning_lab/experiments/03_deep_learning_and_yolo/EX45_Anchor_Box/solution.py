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
    # 1. Calculate bounding box center (bx, by) in grid space using sigmoid adjustment
    sigmoid_tx = 1.0 / (1.0 + np.exp(-tx))
    sigmoid_ty = 1.0 / (1.0 + np.exp(-ty))
    
    bx = sigmoid_tx + grid_x
    by = sigmoid_ty + grid_y
    
    # 2. Calculate bounding box width and height (bw, bh) in image space
    bw = anchor_w * np.exp(tw)
    bh = anchor_h * np.exp(th)
    
    # 3. Scale bx, by to image space using the stride
    bx_img = bx * stride
    by_img = by * stride
    
    # 4. Convert from [cx, cy, w, h] to [x_min, y_min, x_max, y_max] coordinates
    x_min = bx_img - bw / 2
    y_min = by_img - bh / 2
    x_max = bx_img + bw / 2
    y_max = by_img + bh / 2
    
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
    # 1. Calculate the center point of the cell in image space
    cx = (grid_x + 0.5) * stride
    cy = (grid_y + 0.5) * stride
    
    # 2. Calculate boundary coordinates in image space by applying left, top, right, bottom offsets
    x_min = cx - l * stride
    y_min = cy - t * stride
    x_max = cx + r * stride
    y_max = cy + b * stride
    
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
    print(f"Decoded Anchor-Based Box: {[round(float(coord), 2) for coord in box_anchor_based]} (Expected: [142.23, 120.44, 212.96, 225.24])")
    print(f"Decoded Anchor-Free Box:  {[round(float(coord), 2) for coord in box_anchor_free]} (Expected: [137.6, 118.4, 224.0, 246.4])")
