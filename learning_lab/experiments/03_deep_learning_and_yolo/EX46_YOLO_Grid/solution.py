import numpy as np

def assign_grid_cell(bbox, image_size=(640, 640)):
    """
    Simulates YOLO assignment of an object bounding box to the correct grid scale and cell.
    
    Args:
        bbox (list or np.ndarray): [x_min, y_min, x_max, y_max] in image space.
        image_size (tuple): Width, height of the input image.
        
    Returns:
        dict: Containing stride, grid size, cell index (col, row), and center coordinates.
    """
    x1, y1, x2, y2 = bbox
    
    # 1. Calculate width, height, and area of the bounding box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    
    # 2. Calculate center point of the bounding box (cx, cy)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    # 3. Determine the most appropriate stride based on object area
    # - Small objects (area < 64^2): Stride 8 (80x80 grid)
    # - Medium objects (64^2 <= area < 192^2): Stride 16 (40x40 grid)
    # - Large objects (area >= 192^2): Stride 32 (20x20 grid)
    if area < 64 ** 2:
        stride = 8
    elif area < 192 ** 2:
        stride = 16
    else:
        stride = 32
    
    # 4. Calculate grid cell indices (col, row) where the object center falls
    cell_col = int(cx // stride)
    cell_row = int(cy // stride)
    
    # Grid dimensions (columns, rows)
    grid_w = image_size[0] // stride
    grid_h = image_size[1] // stride
    
    return {
        "stride": stride,
        "grid_shape": (grid_w, grid_h),
        "cell_index": (cell_col, cell_row),
        "center_pixel": (cx, cy)
    }

if __name__ == "__main__":
    # Test cases: different components from the PTT dataset
    test_objects = {
        "small-valve": [100.0, 100.0, 140.0, 130.0],  # Small object (area: 1200)
        "control-valve": [200.0, 200.0, 310.0, 320.0], # Medium object (area: 13200)
        "wellhead": [150.0, 150.0, 480.0, 520.0]       # Large object (area: 122100)
    }
    
    print("--- Grid Assignment Results ---")
    for name, bbox in test_objects.items():
        assignment = assign_grid_cell(bbox)
        print(f"Object: {name}")
        print(f"  Bbox: {bbox}")
        print(f"  Center: {assignment['center_pixel']}")
        print(f"  Assigned Stride: {assignment['stride']} (Grid: {assignment['grid_shape']})")
        print(f"  Responsible Cell: col={assignment['cell_index'][0]}, row={assignment['cell_index'][1]}")
        print()
