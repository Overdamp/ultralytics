import os
import glob
from collections import defaultdict
import json
import yaml

dataset_path = "/home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11"
data_yaml_path = os.path.join(dataset_path, "data.yaml")

with open(data_yaml_path, 'r') as f:
    data_cfg = yaml.safe_load(f)

classes = data_cfg.get('names', [])
nc = data_cfg.get('nc', 0)

splits = {
    'train': {
        'images': os.path.join(dataset_path, 'train/images'),
        'labels': os.path.join(dataset_path, 'train/labels')
    },
    'val': {
        'images': os.path.join(dataset_path, 'valid/images'),
        'labels': os.path.join(dataset_path, 'valid/labels')
    },
    'test': {
        'images': os.path.join(dataset_path, 'test/images'),
        'labels': os.path.join(dataset_path, 'test/labels')
    }
}

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.PNG', '.JPG', '.JPEG')

report = {}

for split_name, paths in splits.items():
    img_dir = paths['images']
    lbl_dir = paths['labels']
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        continue
        
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(image_extensions)])
    lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
    
    img_basenames = {os.path.splitext(f)[0] for f in img_files}
    lbl_basenames = {os.path.splitext(f)[0] for f in lbl_files}
    
    images_without_labels = img_basenames - lbl_basenames
    labels_without_images = lbl_basenames - img_basenames
    
    empty_labels = []
    corrupt_labels = []
    out_of_bounds_coords = []
    invalid_class_ids = []
    
    bbox_counts = defaultdict(int)
    total_bboxes = 0
    
    for lbl_file in lbl_files:
        lbl_path = os.path.join(lbl_dir, lbl_file)
        if os.path.getsize(lbl_path) == 0:
            empty_labels.append(lbl_file)
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            empty_labels.append(lbl_file)
            continue
            
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                corrupt_labels.append(f"{lbl_file}:{line_idx}")
                continue
                
            try:
                class_idx = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                corrupt_labels.append(f"{lbl_file}:{line_idx}")
                continue
                
            if class_idx < 0 or class_idx >= nc:
                invalid_class_ids.append(f"{lbl_file}:{line_idx}:{class_idx}")
            else:
                bbox_counts[class_idx] += 1
                total_bboxes += 1
                
            for c_val in coords:
                if c_val < 0.0 or c_val > 1.0:
                    out_of_bounds_coords.append(f"{lbl_file}:{line_idx}")
                    break
                    
    report[split_name] = {
        'num_images': len(img_files),
        'num_labels': len(lbl_files),
        'images_without_labels': list(images_without_labels),
        'labels_without_images': list(labels_without_images),
        'empty_labels': empty_labels,
        'corrupt_labels': corrupt_labels,
        'out_of_bounds_coords': out_of_bounds_coords,
        'invalid_class_ids': invalid_class_ids,
        'bbox_counts': dict(bbox_counts),
        'total_bboxes': total_bboxes
    }

print(json.dumps({
    'classes': classes,
    'nc': nc,
    'splits_data': report
}, indent=2))
