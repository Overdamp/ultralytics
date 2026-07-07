# 🧠 EX47: คะแนนความมั่นใจในการตรวจจับวัตถุ (Confidence Score in Object Detection)

คะแนนความมั่นใจ (confidence score) คือตัววัดค่าความน่าจะเป็นที่ระบุว่า bounding box ที่ทำนายนั้นมีความเป็นไปได้มากเพียงใดที่จะมีวัตถุอยู่ภายใน และกล่องนั้นครอบคลุมวัตถุได้แม่นยำเพียงใด

---

## 1. สูตรทางคณิตศาสตร์ (Mathematical Formulation)

การคำนวณคะแนนความมั่นใจได้รับการพัฒนาปรับปรุงในแต่ละรุ่นของ YOLO ดังนี้:

### A. YOLO รุ่นดั้งเดิม (v1 - v5)
ในสถาปัตยกรรมแบบอิงแองเคอร์รุ่นดั้งเดิม คะแนนความมั่นใจสำหรับ bounding box จะถูกกำหนดสูตรดังนี้:

$$\text{Confidence} = P(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$$

เมื่อ:
*   $P(\text{Object}) \in [0, 1]$ คือ **คะแนนการมีอยู่ของวัตถุ (objectness score)** (ความน่าจะเป็นที่กริดเซลล์นั้นๆ จะมีวัตถุอยู่จริง)
*   $\text{IoU}_{\text{pred}}^{\text{truth}}$ คือค่าการทับซ้อนกันระหว่าง bounding box ที่ทำนายกับ bounding box ที่เป็นจริง

คะแนนความมั่นใจสำหรับแต่ละคลาส (class-specific confidence score) สุดท้ายคำนวณได้จาก:
$$\text{Class Score} = P(\text{Class}_i | \text{Object}) \times \text{Confidence} = P(\text{Class}_i | \text{Object}) \times P(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$$

### B. YOLO แบบไร้แองเคอร์ในปัจจุบัน (v8 - v11)
สถาปัตยกรรม YOLO แบบไร้แองเคอร์ในปัจจุบันได้ตัดส่วนการคำนวณการมีอยู่ของวัตถุ (objectness branch) ออกไปเพื่อลดความหน่วง (latency) ในการประมวลผล แต่จะส่งเอาต์พุตเป็นความน่าจะเป็นของคลาสโดยตรง และคะแนนความมั่นใจจะถูกคำนวณในระหว่างการปรับปรุงประสิทธิภาพ (optimization) โดยใช้ **ตัวจัดสรรตามแนวทางของงาน (Task-Aligned Assigner)** ซึ่งจะรวมคุณภาพของการจำแนกประเภทและการระบุตำแหน่งเข้าไว้ด้วยกันเป็นคะแนนเดียว:

$$t = s^\alpha \times \text{IoU}^\beta$$

เมื่อ:
*   $s$ คือความน่าจะเป็นของคลาสที่ทำนายได้ (predicted class probability)
*   $\text{IoU}$ คือค่าการทับซ้อนระหว่างผลการทำนายและกล่องที่เป็นจริง (ground-truth)
*   $\alpha$ และ $\beta$ คือตัวคูณน้ำหนัก (weighting factors)

---

## 🔬 กรณีศึกษาคอมพิวเตอร์วิทัศน์: การปรับจูนพารามิเตอร์ `conf`
เมื่อทำการรันการอนุมาน (inference) บนชุดข้อมูล PTT การปรับเกณฑ์คะแนนความมั่นใจ (`conf`) จะส่งผลอย่างมากต่อประสิทธิภาพการนำโมเดลไปใช้งานในสถานการณ์จริง

### สถานการณ์สมมติ: การรันการอนุมานด้วยเกณฑ์ที่แตกต่างกัน (Scenario: Running Inference with different thresholds)

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train-3/weights/best.pt')

# Scenario A: High confidence threshold (High Precision, Low Recall)
# Only predicts objects it is 60% sure about. Excellent for avoiding false alarms.
# May miss smaller valves (e.g. 'small-valve').
results_a = model('datasets/coco8/overall-ptt-object-detection.v11i.yolov11/test/images', conf=0.6)

# Scenario B: Low confidence threshold (Low Precision, High Recall)
# Predicts everything down to 15% certainty. Good for picking up every object.
# Will introduce false positives (clutter labeled as 'flange' or 'valve').
results_b = model('datasets/coco8/overall-ptt-object-detection.v11i.yolov11/test/images', conf=0.15)
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX46_YOLO_Grid_TH\|EX46: ระบบกริดของ YOLO]]
*   [[EX48_mAP_TH\|EX48: ค่าเฉลี่ยความแม่นยำเฉลี่ย (Mean Average Precision - mAP)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
