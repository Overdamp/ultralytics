# 🧠 EX42: การหาจุดร่วมและจุดต่าง (Intersection over Union - IoU)

Intersection over Union (IoU) คือตัววัดค่าที่ใช้วัดการทับซ้อนกันระหว่าง bounding box สองกล่อง ได้แก่ bounding box ที่ทำนาย ($A$) และ bounding box ที่เป็นจริงหรือกราวด์ทรูธ (ground truth) ($B$)

---

## 1. สูตรทางคณิตศาสตร์ (Mathematical Formulation)

กำหนดให้กรอบสองกล่องอยู่ในรูปแบบ XYXY:
*   กล่อง A (Box A): $[x_1^A, y_1^A, x_2^A, y_2^A]$
*   กล่อง B (Box B): $[x_1^B, y_1^B, x_2^B, y_2^B]$

### ขั้นตอนที่ 1: พื้นที่ส่วนที่ทับซ้อน (Area of Intersection)
การทับซ้อน (intersection) คือพื้นที่รูปสี่เหลี่ยมผืนผ้าที่ซ้อนทับกัน เราสามารถคำนวณพิกัดของพื้นที่นี้ได้โดยการเลือกค่าที่มากที่สุด (maximum) ของมุมบนซ้าย และเลือกค่าที่น้อยที่สุด (minimum) ของมุมล่างขวา:

$$x_{\min}^{\text{inter}} = \max(x_1^A, x_1^B), \quad y_{\min}^{\text{inter}} = \max(y_1^A, y_1^B)$$
$$x_{\max}^{\text{inter}} = \min(x_2^A, x_2^B), \quad y_{\max}^{\text{inter}} = \min(y_2^A, y_2^B)$$

ความกว้าง ($w$) และความสูง ($h$) ของพื้นที่ทับซ้อนคือ:
$$w_{\text{inter}} = \max(0, x_{\max}^{\text{inter}} - x_{\min}^{\text{inter}})$$
$$h_{\text{inter}} = \max(0, y_{\max}^{\text{inter}} - y_{\min}^{\text{inter}})$$

$$\text{Area}_{\text{inter}} = w_{\text{inter}} \times h_{\text{inter}}$$

### ขั้นตอนที่ 2: พื้นที่ส่วนที่รวมกัน (Area of Union)
พื้นที่รวม (union area) เกิดจากการรวมพื้นที่ของกล่องทั้งสองเข้าด้วยกัน แล้วหักพื้นที่ส่วนที่ทับซ้อนกันออก (เพื่อหลีกเลี่ยงการคำนวณซ้ำสองครั้ง):

$$\text{Area}_A = (x_2^A - x_1^A) \times (y_2^A - y_1^A)$$
$$\text{Area}_B = (x_2^B - x_1^B) \times (y_2^B - y_1^B)$$
$$\text{Area}_{\text{union}} = \text{Area}_A + \text{Area}_B - \text{Area}_{\text{inter}}$$

### ขั้นตอนที่ 3: คำนวณ IoU
$$\text{IoU} = \frac{\text{Area}_{\text{inter}}}{\text{Area}_{\text{union}}}$$

---

## 2. ตัวแปรเสริมประสิทธิภาพของ IoU (YOLO Loss Functions)
IoU มาตรฐานมีข้อเสียที่สำคัญคือ: หากกล่องทั้งสองไม่มีการทับซ้อนกันเลย พื้นที่ทับซ้อนจะเป็น $0$ ซึ่งหมายความว่า $\text{IoU} = 0$ ส่งผลให้ฟังก์ชันการสูญเสีย (loss function) มีลักษณะเป็นระนาบแบนและมีค่าคงที่ ทำให้กระบวนการไล่ระดับความชัน (gradient descent) ไม่สามารถอัปเดตพิกัดของกล่องได้

เพื่อแก้ปัญหานี้ จึงได้มีการนำเสนอรูปแบบ IoU ที่ได้รับการปรับปรุงขั้นสูง ได้แก่:
*   **GIoU (Generalized IoU):** เพิ่มเทอมบทลงโทษ (penalty term) โดยอิงตามพื้นที่ของกล่องนูนที่เล็กที่สุด (smallest enclosing convex box) $C$ ที่ครอบคลุมทั้งกล่อง $A$ และ $B$ เพื่อช่วยให้ความชัน (gradients) ยังสามารถไหลผ่านได้แม้ว่าจะไม่มีส่วนทับซ้อนกันก็ตาม
*   **DIoU (Distance IoU):** ลงโทษตามระยะห่างแบบนอร์มัลไลซ์ระหว่างจุดศูนย์กลางของกล่องทั้งสองใบ ซึ่งช่วยให้กล่องต่างๆ ลู่เข้าหากัน (converge) ได้เร็วขึ้นอย่างมาก
*   **CIoU (Complete IoU - ใช้ใน YOLO):** เพิ่มเทอมบทลงโทษเรื่องอัตราส่วนภาพ (aspect ratio) โดยวัดปัจจัยทางเรขาคณิตหลัก 3 ประการ ได้แก่ พื้นที่ที่ทับซ้อนกัน, ระยะห่างระหว่างจุดศูนย์กลาง, และความสอดคล้องของอัตราส่วนภาพ

---

## 💻 การเขียนโปรแกรมด้วย Python (NumPy)

```python
import numpy as np

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    x1_inter = max(boxA[0], boxB[0])
    y1_inter = max(boxA[1], boxB[1])
    x2_inter = min(boxA[2], boxB[2])
    y2_inter = min(boxA[3], boxB[3])
    
    # Calculate intersection area
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    
    # Calculate union area
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_union = areaA + areaB - area_inter
    
    # Compute IoU
    if area_union == 0:
        return 0.0
    return area_inter / area_union

# Example usage:
box_pred = [100, 100, 210, 210]
box_truth = [120, 120, 220, 220]
print(f"IoU: {calculate_iou(box_pred, box_truth):.4f}")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX41_Bounding_Box_TH\|EX41: ระบบพิกัด Bounding Box]]
*   [[EX43_NMS_TH\|EX43: การตัดทิ้งที่ไม่ใช่จุดสูงสุด (Non-Maximum Suppression - NMS)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
