import numpy as np

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in XYXY format:
    [x1, y1, x2, y2]
    """
    # 1. Determine intersection coordinates
    x1_inter = max(boxA[0], boxB[0])
    y1_inter = max(boxA[1], boxB[1])
    x2_inter = min(boxA[2], boxB[2])
    y2_inter = min(boxA[3], boxB[3])
    
    # 2. Calculate intersection area
    width_inter = max(0.0, x2_inter - x1_inter)
    height_inter = max(0.0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    
    # 3. Calculate individual box areas
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # 4. Calculate union area
    area_union = areaA + areaB - area_inter
    
    # 5. Compute IoU
    if area_union == 0.0:
        return 0.0
        
    return float(area_inter / area_union)

if __name__ == "__main__":
    box_truth = [100.0, 100.0, 200.0, 200.0]
    box_pred = [120.0, 120.0, 220.0, 220.0]
    
    iou = calculate_iou(box_truth, box_pred)
    
    print("--- Training Results ---")
    print(f"Ground Truth : {box_truth}")
    print(f"Prediction   : {box_pred}")
    print(f"IoU          : {iou:.6f}")
