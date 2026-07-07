# 🧠 EX49: การวิเคราะห์ผลลัพธ์ของ YOLO (YOLO Result Analysis)

การวิเคราะห์ผลลัพธ์การฝึกสอนของ YOLO คือกระบวนการตรวจสอบค่าชี้วัดเอาต์พุต (output metrics) กราฟเส้นโค้งประสิทธิภาพ (curves) และรูปภาพการทำนาย เพื่อวินิจฉัยประสิทธิภาพการทำงานของโมเดล ตรวจจับปัญหาการเรียนรู้จำเพาะเกินไป (overfitting) และวางแผนปรับแต่งพารามิเตอร์ต่างๆ

---

## 1. โครงสร้างของไดเรกทอรีเอาต์พุตการฝึกสอนของ YOLO

เมื่อคุณฝึกสอนโมเดล YOLO ไดเรกทอรีจะถูกสร้างขึ้นภายใต้ `runs/detect/train*/` ซึ่งบรรจุไฟล์ต่างๆ ดังนี้:

| ชื่อไฟล์ (File Name) | คำอธิบาย (Description) | สิ่งที่ต้องสังเกต (What to Look For) |
| :--- | :--- | :--- |
| **`weights/best.pt`** | สถานะน้ำหนักของโมเดล (weights) ที่ได้ค่า mAP ของการตรวจสอบข้อมูล (validation) สูงสุด | ใช้ไฟล์นี้สำหรับการนำไปใช้งานจริง (deployment) และทดสอบการอนุมาน (inference testing) |
| **`weights/last.pt`** | สถานะน้ำหนักของโมเดลจากรอบการเทรน (epoch) สุดท้าย | ใช้ไฟล์นี้หากคุณต้องการฝึกสอนต่อจากจุดที่หยุดลง |
| **`results.csv`** | บันทึกประวัติแบบแถวต่อแถวของทุกรอบการเทรน | ติดตามเวลาของแต่ละรอบ, ค่าความสูญเสีย (losses: box, class, dfl), อัตราการเรียนรู้ (learning rates), และตัววัดประสิทธิภาพการตรวจสอบข้อมูล |
| **`results.png`** | แผนภูมิพล็อตกราฟของค่าความสูญเสียและตัววัดประสิทธิภาพทั้งหมดตลอดระยะเวลาแต่ละรอบ | มองหากราฟค่าความสูญเสียที่ลดลงอย่างสม่ำเสมอและกราฟ mAP ที่ไต่ระดับขึ้นสูง |
| **`confusion_matrix_normalized.png`** | ตารางกริดแสดงการทำนายที่ถูกต้องบนแนวเส้นทแยงมุมหลัก และความสับสนระหว่างคลาสที่นอกแนวเส้นทแยงมุม | มองหาค่าบนแนวเส้นทแยงมุมหลักที่เข้าใกล้ `1.0` ค่าที่สูงนอกแนวเส้นทแยงมุมหลักแสดงถึงความสับสนระหว่างคลาส |
| **`BoxPR_curve.png`** | เส้นโค้งความแม่นยำและความระลึก (Precision-Recall curve) ที่พล็อตสำหรับทุกคลาส | แสดงการแลกเปลี่ยนประสิทธิภาพ (trade-off) เส้นโค้งที่สมบูรณ์แบบจะโค้งนูนออกไปทางมุมบนขวา |
| **`val_batch*_pred.jpg`** | การแสดงผลการทำนาย bounding box เชิงภาพบนรูปภาพที่ใช้ตรวจสอบข้อมูล | ตรวจสอบด้วยตาเพื่อหาการคลาดเคลื่อนของกล่อง, วัตถุที่ตรวจจับไม่เจอ หรือการตรวจจับซ้ำซ้อน |

---

## 2. การวินิจฉัยพฤติกรรมการฝึกสอน (Diagnosing Training Behavior)

### A. ปัญหาการเรียนรู้จำเพาะเกินไป (Overfitting)
*   **อาการ (Symptom):** ค่าความสูญเสียจากการเทรน (`train/box_loss`, `train/cls_loss`) ยังคงลดลงอย่างต่อเนื่อง แต่ค่าความสูญเสียจากการตรวจสอบข้อมูล (`val/box_loss`, `val/cls_loss`) เริ่มสูงขึ้นหรือราบเรียบไม่ลดลงต่อ
*   **แนวทางแก้ไข (Solution):** เพิ่มเทอมปรับลดทอนค่าน้ำหนัก (regularization: `weight_decay`), ปรับระดับข้อมูลเสริม (augmentations: `mosaic`, `mixup`), หรือหยุดการเทรนก่อนกำหนดโดยใช้การตั้งค่า `epochs` หรือ `patience` ที่น้อยลง

### B. ปัญหาการเรียนรู้ไม่เพียงพอ (Underfitting)
*   **อาการ (Symptom):** ค่าความสูญเสียทั้งจากการเทรนและการตรวจสอบข้อมูลยังคงอยู่ในระดับสูงหรือคงที่อย่างรวดเร็ว และค่า mAP ยังคงอยู่ในระดับต่ำ
*   **แนวทางแก้ไข (Solution):** เทรนจำนวนรอบ (epochs) ให้มากขึ้น, ใช้ขนาดโมเดลที่ใหญ่ขึ้น (เช่น ปรับขนาดจาก `yolo26n` ขึ้นเป็น `yolo26s`), หรือลดค่าน้ำหนักของ regularization ลง

---

## 3. การพล็อตกราฟความก้าวหน้าของการเทรนด้วย Pandas

คุณสามารถแยกวิเคราะห์ไฟล์ `results.csv` โดยใช้ Python เพื่อจำลองเส้นโค้งแสดงค่าความสูญเสียของการฝึกสอนและตรวจสอบข้อมูลออกมาได้:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training logs
df = pd.read_csv('runs/detect/train-3/results.csv')

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()

# Plot Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Box Loss')
plt.legend()
plt.grid(True)
plt.savefig('learning_lab/experiments/03_deep_learning_and_yolo/EX49_YOLO_Result_Analysis/loss_curve_check.png')
print("Loss curve check chart saved successfully.")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX48_mAP_TH\|EX48: ค่าเฉลี่ยความแม่นยำเฉลี่ย (Mean Average Precision - mAP)]]
*   [[EX50_Hyperparameter_Tuning_TH\|EX50: การปรับจูนไฮเปอร์พารามิเตอร์ (Hyperparameter Tuning)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
