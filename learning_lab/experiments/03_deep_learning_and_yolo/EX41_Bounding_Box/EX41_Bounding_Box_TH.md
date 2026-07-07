# 🧠 EX41: ระบบพิกัด Bounding Box (Bounding Box Coordinate Systems)

Bounding box (กรอบล้อมรอบวัตถุ) ใช้ในการกำหนดตำแหน่งเชิงพื้นที่และขนาดของวัตถุในรูปภาพ การทำความเข้าใจเกี่ยวกับวิธีการแทนค่า การแปลงค่า และการทำให้พิกัดเหล่านี้เป็นบรรทัดฐาน (normalization) เป็นสิ่งสำคัญอย่างยิ่งในการจัดการกระบวนการประมวลผลข้อมูล (pipeline) สำหรับการตรวจจับวัตถุ (object detection)

---

## 1. รูปแบบการแทนพิกัด (Coordinate Representation Formats)

มีสองรูปแบบหลักในการแทนพิกัด bounding box ที่ใช้ในคอมพิวเตอร์วิทัศน์ (computer vision):

### A. รูปแบบมุม (Corners Format - XYXY)
*   **การแทนค่า (Representation):** $[x_1, y_1, x_2, y_2]$
*   **คำอธิบาย (Description):** พิกัดของมุมบนซ้าย $(x_1, y_1)$ และมุมล่างขวา $(x_2, y_2)$
*   **กรณีการใช้งาน (Use Case):** รูปแบบเริ่มต้นสำหรับการวาดกรอบ (bounding box ใน OpenCV) และใช้ในการคำนวณตัววัดการทับซ้อน เช่น IoU

### B. รูปแบบจุดศูนย์กลาง (Centroid Format - XYWH)
*   **การแทนค่า (Representation):** $[x_c, y_c, w, h]$
*   **คำอธิบาย (Description):** พิกัดของจุดศูนย์กลางของกรอบ $(x_c, y_c)$ ร่วมกับความกว้าง ($w$) และความสูง ($h$) ของกรอบ
*   **กรณีการใช้งาน (Use Case):** รูปแบบหลักที่ทำนายโดยส่วนหัว (head) ของโครงข่ายประสาทเทียม

---

## 2. คณิตศาสตร์ในการแปลงค่า Bounding Box

### การแปลงจาก XYWH เป็น XYXY:
กำหนดให้กรอบคือ $[x_c, y_c, w, h]$:

$$x_1 = x_c - \frac{w}{2}, \quad y_1 = y_c - \frac{h}{2}$$
$$x_2 = x_c + \frac{w}{2}, \quad y_2 = y_c + \frac{h}{2}$$

### การแปลงจาก XYXY เป็น XYWH:
กำหนดให้กรอบคือ $[x_1, y_1, x_2, y_2]$:

$$x_c = \frac{x_1 + x_2}{2}, \quad y_c = \frac{y_1 + y_2}{2}$$
$$w = x_2 - x_1, \quad h = y_2 - y_1$$

---

## 3. รูปแบบคำอธิบายประกอบแบบนอร์มัลไลซ์ของ YOLO (YOLO Normalized Annotation Format)
เพื่อให้แน่ใจว่าป้ายกำกับ (labels) ของ bounding box มีคุณสมบัติ**ไม่เปลี่ยนแปรตามขนาด (scale-invariant)** (สามารถใช้กับรูปภาพที่มีขนาดต่างกันได้โดยที่พิกัดไม่เพี้ยน) YOLO จึงจัดเก็บคำอธิบายประกอบให้อยู่ในรูปแบบนอร์มัลไลซ์ (normalized) ระหว่าง `0` ถึง `1`:

เมื่อกำหนดให้รูปภาพมีความกว้าง $W$ และความสูง $H$ พิกัดจะถูกนอร์มัลไลซ์ดังนี้:

$$x_{\text{norm}} = \frac{x_c}{W}, \quad y_{\text{norm}} = \frac{y_c}{H}$$
$$w_{\text{norm}} = \frac{w}{W}, \quad h_{\text{norm}} = \frac{h}{H}$$

ในไฟล์ข้อความของป้ายกำกับที่ตรงกับแต่ละรูปภาพ YOLO คาดหวังรูปแบบดังนี้:
```text
<class_id> <x_norm> <y_norm> <w_norm> <h_norm>
```

---

## 💻 ฟังก์ชันการแปลงพิกัดใน Python (NumPy)

```python
import numpy as np

def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from [xc, yc, w, h] to [x1, y1, x2, y2].
    Can handle single box list or a NumPy array of shape (N, 4).
    """
    boxes = np.array(boxes)
    converted = np.zeros_like(boxes)
    converted[..., 0] = boxes[..., 0] - (boxes[..., 2] / 2) # x1
    converted[..., 1] = boxes[..., 1] - (boxes[..., 3] / 2) # y1
    converted[..., 2] = boxes[..., 0] + (boxes[..., 2] / 2) # x2
    converted[..., 3] = boxes[..., 1] + (boxes[..., 3] / 2) # y2
    return converted

def xyxy_to_xywh(boxes):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [xc, yc, w, h].
    """
    boxes = np.array(boxes)
    converted = np.zeros_like(boxes)
    converted[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2 # xc
    converted[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2 # yc
    converted[..., 2] = boxes[..., 2] - boxes[..., 0]       # w
    converted[..., 3] = boxes[..., 3] - boxes[..., 1]       # h
    return converted
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX40_Embeddings_TH\|EX40: ฟีเจอร์เอ็มเบดดิง (Feature Embeddings)]]
*   [[EX42_IoU_TH\|EX42: การหาจุดร่วมและจุดต่าง (Intersection over Union - IoU)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
