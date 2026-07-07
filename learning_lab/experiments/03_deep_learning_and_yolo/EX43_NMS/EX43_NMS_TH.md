# 🧠 EX43: การตัดทิ้งที่ไม่ใช่จุดสูงสุด (Non-Maximum Suppression - NMS)

Non-Maximum Suppression (NMS) คืออัลกอริทึมสำหรับขั้นตอนหลังการประมวลผล (post-processing) ที่ใช้ในการตรวจจับวัตถุ (object detection) เพื่อกำจัด bounding box ที่ซ้ำซ้อนและทับซ้อนกันซึ่งถูกทำนายสำหรับวัตถุชิ้นเดียวกัน โดยจะเหลือไว้เพียงกล่องทำนายที่ดีที่สุดเพียงกล่องเดียวเท่านั้น

---

## 1. ทำไม NMS จึงมีความจำเป็น
ตัวตรวจจับวัตถุแบบขั้นตอนเดียว (Single-stage object detectors) เช่น YOLO จะทำการประเมินกริดเซลล์ (grid cells) นับพันเซลล์ในขนาดสเกลต่างๆ บ่อยครั้งที่เซลล์หลายๆ เซลล์รอบวัตถุจะทำนายค่าคะแนนความมั่นใจ (confidence score) ที่สูงสำหรับวัตถุชิ้นเดียวกัน ส่งผลให้เกิดกล่องทับซ้อนกันหลายใบ NMS จะทำหน้าที่คัดกรองกล่องเหล่านี้ออก เพื่อให้แน่ใจว่าวัตถุแต่ละชิ้นจะมีกล่องทำนายเพียงกล่องเดียวพอดี

---

## 2. อัลกอริทึม NMS

### อินพุต (Inputs):
*   $B = \{b_1, \dots, b_n\}$: กล่อง bounding box ที่มีสิทธิ์ได้รับเลือก (candidate bounding boxes)
*   $S = \{s_1, \dots, s_n\}$: คะแนนความมั่นใจในการทำนายที่สอดคล้องกัน (corresponding prediction confidence scores)
*   $N_t$: เกณฑ์ขีดเริ่มเปลี่ยนของ IoU (IoU threshold) (โดยทั่วไปจะอยู่ระหว่าง `0.45` ถึง `0.7`)

### ขั้นตอน (Steps):
1.  เริ่มต้นสร้างรายการว่างสำหรับผลการตรวจจับที่ได้รับเลือก $D = \{\}$
2.  ในขณะที่ $B$ ยังไม่ว่าง:
    *   ค้นหากล่อง $M$ ที่มีคะแนนความมั่นใจสูงสุดใน $S$
    *   นำ $M$ ออกจาก $B$ และเพิ่มเข้าไปใน $D$
    *   สำหรับทุกกล่อง $b_i$ ที่เหลืออยู่ใน $B$ ให้คำนวณการทับซ้อน $\text{IoU}(M, b_i)$
    *   หาก $\text{IoU}(M, b_i) \ge N_t$ ให้ทิ้งกล่อง $b_i$ และคะแนน $s_i$ ออกจาก $B$ และ $S$
3.  คืนค่ารายการตรวจจับสุดท้าย $D$

---

## 3. ตัวแปรปรับปรุงระดับสูง: Soft-NMS
ในอัลกอริทึม NMS มาตรฐานแบบดั้งเดิม (hard NMS) หากมีวัตถุชิ้นที่สองอยู่จริงและซ้อนทับกับวัตถุชิ้นแรก (เช่น วาล์วสองตัวที่ติดตั้งอยู่ข้างกันบนท่อร่วม) hard NMS จะลบกล่องของวัตถุชิ้นที่สองทิ้งไปอย่างสิ้นเชิง ส่งผลให้เกิดข้อผิดพลาดแบบ **ผลลบเท็จ (False Negative)**

**Soft-NMS** แก้ปัญหานี้โดยการลดทอน (decaying) คะแนนความมั่นใจของกล่องที่ทับซ้อนกันแทนที่จะลบทิ้งไปทันที หากคะแนนที่ลดทอนแล้วยังคงสูงกว่าเกณฑ์ขีดเริ่มเปลี่ยนความมั่นใจ กล่องนั้นก็จะยังคงถูกรักษาไว้:

$$s_i \leftarrow s_i \cdot \exp\left( -\frac{\text{IoU}(M, b_i)^2}{\sigma} \right)$$

---

## 💻 การเขียนโปรแกรมด้วย Python (NumPy)

นี่คือการเขียนโปรแกรม NMS ที่ประมวลผลอย่างรวดเร็วด้วย NumPy สำหรับพิกัดแบบ 1 มิติ:

```python
import numpy as np

def nms_numpy(boxes, scores, iou_threshold):
    # Coordinates of boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort boxes by confidence score descending
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate overlap coordinates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Calculate overlap area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        # Compute IoU
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Keep boxes where IoU is less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX42_IoU_TH\|EX42: การหาจุดร่วมและจุดต่าง (Intersection over Union - IoU)]]
*   [[EX44_Object_Detection_Pipeline_TH\|EX44: ขั้นตอนการประมวลผลสำหรับการตรวจจับวัตถุ (Object Detection Pipeline)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
