import numpy as np

class ObjectDetectionPipeline:
    def __init__(self, image_size=(640, 640), num_classes=3, iou_threshold=0.5, score_threshold=0.6):
        self.image_size = image_size
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def backbone(self, x):
        """
        Stage 1: Backbone (Feature Extractor)
        Extracts multiscale feature maps P3, P4, P5 with strides 8, 16, 32.
        P3 shape: (80, 80, 128)
        P4 shape: (40, 40, 256)
        P5 shape: (20, 20, 512)
        """
        # Calculate dimensions for P3, P4, and P5 based on strides 8, 16, and 32
        p3_h = self.image_size[0] // 8
        p3_w = self.image_size[1] // 8
        p4_h = self.image_size[0] // 16
        p4_w = self.image_size[1] // 16
        p5_h = self.image_size[0] // 32
        p5_w = self.image_size[1] // 32
        
        # Simulating outputs
        p3 = np.random.randn(p3_h, p3_w, 128)
        p4 = np.random.randn(p4_h, p4_w, 256)
        p5 = np.random.randn(p5_h, p5_w, 512)
        
        return {"P3": p3, "P4": p4, "P5": p5}

    def neck(self, features):
        """
        Stage 2: Neck (Feature Aggregator)
        Performs FPN top-down fusion.
        Upsamples P5 and merges with P4; upsamples the merged P4 and merges with P3.
        """
        p3 = features["P3"]
        p4 = features["P4"]
        p5 = features["P5"]
        
        # Upsample P5 to match P4 (2x upsampling)
        p5_upsampled = np.repeat(np.repeat(p5, 2, axis=0), 2, axis=1)
        
        # Merge features by adding lateral P4 to upsampled P5 (project channel depth to match)
        f4 = p4 + p5_upsampled[:, :, :256]
        
        # Upsample f4 to match P3 (2x upsampling)
        f4_upsampled = np.repeat(np.repeat(f4, 2, axis=0), 2, axis=1)
        
        # Merge by adding lateral P3 to upsampled f4 (project channels to match 128 depth)
        f3 = p3 + f4_upsampled[:, :, :128]
        
        return {"F3": f3, "F4": f4, "F5": p5}

    def head(self, fused_features):
        """
        Stage 3: Decoupled Head (Predictor)
        Generates classification logits and bounding box coordinate predictions.
        """
        candidate_boxes = []
        candidate_scores = []
        candidate_class_ids = []
        
        for scale_name, feat in fused_features.items():
            H, W, C = feat.shape
            stride = self.image_size[0] // H
            
            # Predict scores for each class (sigmoid output shape: H x W x num_classes)
            cls_logits = np.random.uniform(-5.0, 5.0, size=(H, W, self.num_classes))
            scores = 1.0 / (1.0 + np.exp(-cls_logits))  # Sigmoid
            
            # Predict bounding boxes relative to grid cells
            # Bbox coordinate shape: H x W x 4 in XYXY format
            bbox_preds = np.zeros((H, W, 4))
            for y in range(H):
                for x in range(W):
                    # Center of grid cell in image coordinates
                    cx = x * stride + stride / 2
                    cy = y * stride + stride / 2
                    
                    # Simulating width and height of predictions
                    w = np.random.uniform(30.0, 150.0)
                    h = np.random.uniform(30.0, 150.0)
                    
                    bbox_preds[y, x] = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            
            # Flatten and filter predictions by score threshold
            for y in range(H):
                for x in range(W):
                    box_scores = scores[y, x]
                    class_id = np.argmax(box_scores)
                    max_score = box_scores[class_id]
                    
                    if max_score >= self.score_threshold:
                        candidate_boxes.append(bbox_preds[y, x])
                        candidate_scores.append(max_score)
                        candidate_class_ids.append(class_id)
                        
        return np.array(candidate_boxes), np.array(candidate_scores), np.array(candidate_class_ids)

    def nms_postprocess(self, boxes, scores, class_ids):
        """
        Stage 4: Post-Processing (NMS)
        Performs class-wise Non-Maximum Suppression to filter redundant detections.
        """
        if len(boxes) == 0:
            return [], [], []
            
        keep_final = []
        
        for c in range(self.num_classes):
            class_mask = (class_ids == c)
            if not np.any(class_mask):
                continue
                
            c_boxes = boxes[class_mask]
            c_scores = scores[class_mask]
            c_indices = np.where(class_mask)[0]
            
            x1 = c_boxes[:, 0]
            y1 = c_boxes[:, 1]
            x2 = c_boxes[:, 2]
            y2 = c_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            
            order = c_scores.argsort()[::-1]
            c_keep = []
            
            while order.size > 0:
                i = order[0]
                c_keep.append(c_indices[i])
                
                if order.size == 1:
                    break
                    
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                intersection = w * h
                
                union = areas[i] + areas[order[1:]] - intersection
                iou = intersection / union
                
                inds = np.where(iou <= self.iou_threshold)[0]
                order = order[inds + 1]
                
            keep_final.extend(c_keep)
            
        return boxes[keep_final], scores[keep_final], class_ids[keep_final]

    def predict(self, raw_image):
        """
        Full Pipeline prediction flow.
        """
        # 1. Backbone features
        features = self.backbone(raw_image)
        # 2. Neck aggregation
        fused = self.neck(features)
        # 3. Head predictions
        boxes, scores, class_ids = self.head(fused)
        # 4. Post-processing (NMS)
        final_boxes, final_scores, final_class_ids = self.nms_postprocess(boxes, scores, class_ids)
        
        return final_boxes, final_scores, final_class_ids

if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = ObjectDetectionPipeline(num_classes=3, score_threshold=0.85, iou_threshold=0.5)
    
    # Input dummy image (640x640x3)
    dummy_image = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
    
    # Predict
    boxes, scores, class_ids = pipeline.predict(dummy_image)
    
    print("--- Pipeline Forward Pass Results ---")
    if len(boxes) > 0:
        print(f"Detected {len(boxes)} objects after NMS.")
        print(f"Scores: {[float(f) for f in scores[:5]]}")
        print(f"Class IDs: {[int(c) for c in class_ids[:5]]}")
    else:
        print("Pipeline executed successfully, but no detections passed the high threshold.")
