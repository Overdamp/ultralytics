import numpy as np

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in XYXY format:
    [x1, y1, x2, y2]
    """
    # TODO: Calculate the coordinates of the intersection rectangle
    # Hint: x1_inter = max(boxA[0], boxB[0])
    x1_inter = 0.0
    y1_inter = 0.0
    x2_inter = 0.0
    y2_inter = 0.0
    
    # TODO: Calculate the width, height, and area of the intersection
    # Hint: Use max(0.0, x2_inter - x1_inter) to prevent negative dimensions
    area_inter = 0.0
    
    # TODO: Calculate the areas of Box A and Box B
    areaA = 0.0
    areaB = 0.0
    
    # TODO: Calculate the area of the union
    area_union = 0.0
    
    # TODO: Calculate and return IoU
    iou = 0.0
    
    return float(iou)

if __name__ == "__main__":
    box_truth = [100.0, 100.0, 200.0, 200.0]
    box_pred = [120.0, 120.0, 220.0, 220.0]
    
    iou = calculate_iou(box_truth, box_pred)
    
    print("--- Training Results ---")
    if iou != 0.0:
        print(f"Ground Truth : {box_truth}")
        print(f"Prediction   : {box_pred}")
        print(f"IoU          : {iou:.6f}")
    else:
        print("IoU calculations not implemented yet.")
