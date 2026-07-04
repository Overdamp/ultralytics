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
    
    # TODO: Calculate width, height, and area of the bounding box
    w = 0.0
    h = 0.0
    area = 0.0
    
    # TODO: Calculate center point of the bounding box (cx, cy)
    cx = 0.0
    cy = 0.0
    
    # TODO: Determine the most appropriate stride based on object area
    # - Small objects (area < 64^2): Stride 8 (80x80 grid)
    # - Medium objects (64^2 <= area < 192^2): Stride 16 (40x40 grid)
    # - Large objects (area >= 192^2): Stride 32 (20x20 grid)
    stride = 8
    
    # TODO: Calculate grid cell indices (col, row) where the object center falls
    # Hint: Divide cx and cy by stride and take the floor (cast to int)
    cell_col = 0
    cell_row = 0
    
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
        "small-valve": [100.0, 100.0, 140.0, 130.0],  # Small object
        "control-valve": [200.0, 200.0, 310.0, 320.0], # Medium object
        "wellhead": [150.0, 150.0, 480.0, 520.0]       # Large object
    }
    
    print("--- Grid Assignment Results ---")
    for name, bbox in test_objects.items():
        assignment = assign_grid_cell(bbox)
        if assignment["stride"] == 8 and assignment["cell_index"] == (0, 0):
            print(f"{name} assignment not implemented yet.")
        else:
            print(f"Object: {name}")
            print(f"  Bbox: {bbox}")
            print(f"  Center: {assignment['center_pixel']}")
            print(f"  Assigned Stride: {assignment['stride']} (Grid: {assignment['grid_shape']})")
            print(f"  Responsible Cell: col={assignment['cell_index'][0]}, row={assignment['cell_index'][1]}")
            print()
