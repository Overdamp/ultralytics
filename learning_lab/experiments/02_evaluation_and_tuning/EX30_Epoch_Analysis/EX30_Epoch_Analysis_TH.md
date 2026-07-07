# 🧠 EX30: การวิเคราะห์จำนวนรอบการฝึกสอนและเส้นโค้งความสามารถในการสรุปนัยทั่วไป (Epoch Analysis & Generalization Curves)

Epoch (จำนวนรอบการฝึกสอน) หมายถึงการที่ขั้นตอนวิธีฝึกสอน (Training algorithm) ประมวลผลผ่านชุดข้อมูลฝึกสอนทั้งหมดครบถ้วนหนึ่งรอบ การวิเคราะห์พฤติกรรมของตัวชี้วัดความสูญเสีย (Loss metrics) ในแต่ละ Epoch จึงเป็นสิ่งจำเป็นอย่างยิ่งในการทำความเข้าใจอัตราการเรียนรู้ของโมเดลและป้องกันการเรียนรู้เกิน (Overfitting)

---

## 1. ความแตกต่างระหว่าง Epochs, Batches และ Iterations

ในการวิเคราะห์การฝึกสอน เราต้องเข้าใจความสัมพันธ์ระหว่างสามคำนี้:
*   **Epoch:** การส่งผ่านข้อมูลทั้งหมดในชุดข้อมูลจนครบรอบหนึ่งครั้ง
*   **ขนาดกลุ่มข้อมูล (Batch Size):** จำนวนตัวอย่างข้อมูลฝึกสอนที่ถูกประมวลผลในการส่งผ่านไปข้างหน้าและย้อนกลับหนึ่งครั้ง
*   **การวนซ้ำ (Iteration หรือ Step):** การคำนวณและอัปเดตเกรเดียนต์หนึ่งครั้ง

$$\text{Iterations per Epoch} = \left\lceil \frac{\text{Total Training Samples}}{\text{Batch Size}} \right\rceil$$

### ตัวอย่าง:
หากคุณฝึกสอนโมเดลด้วยชุดข้อมูลที่สร้างขึ้นเองจำนวน $800$ ภาพ โดยใช้ `batch=16`:
$$\text{Iterations per Epoch} = \frac{800}{16} = 50 \text{ steps}$$
การฝึกสอนเป็นจำนวน $100$ epochs จะมีการอัปเดตเกรเดียนต์รวมทั้งสิ้น $5,000$ ครั้ง

---

## 2. เส้นโค้งความสามารถในการสรุปนัยทั่วไปและการวิเคราะห์การเรียนรู้เกิน (Generalization Curves & Overfitting Analysis)
การพล็อตกราฟเปรียบเทียบค่าความสูญเสียจากการฝึกสอน (Training loss) และการทดสอบความถูกต้อง (Validation loss) ในแต่ละ Epoch จะทำให้เราได้ **เส้นโค้งความสามารถในการสรุปนัยทั่วไป (Generalization curves)** ที่ช่วยตรวจวินิจฉัยสุขภาพการเรียนรู้ของโมเดลได้:

```
Loss (ค่าความสูญเสีย)
  |
  |    \                 /  <-- Validation Loss เริ่มสูงขึ้น (Overfitting! / เรียนรู้เกิน)
  |     \               /
  |      \_____________/    <-- จุดหยุดที่เหมาะสมที่สุด (Optimal stopping point)
  |       \           \
  |________\___________\_____
  0                           Epochs (จำนวนรอบ)
```

1.  **การเรียนรู้ต่ำเกินไป (Underfitting):** ทั้งค่าความสูญเสียจากการฝึกสอนและผลการทดสอบความถูกต้องยังคงสูงและราบเรียบ โมเดลยังไม่สามารถเรียนรู้รูปแบบโครงสร้างหลักของข้อมูลได้
2.  **การเรียนรู้เกิน (Overfitting):** ค่าความสูญเสียจากการฝึกสอนยังคงลดลงอย่างต่อเนื่องเข้าใกล้ศูนย์ แต่ค่าความสูญเสียจากผลการทดสอบความถูกต้องกลับคงที่และเริ่มสูงขึ้น แสดงว่าโมเดลกำลังจดจำเฉพาะภาพในชุดข้อมูลฝึกสอน
3.  **การหยุดที่เหมาะสมที่สุด (Early Stopping):** การฝึกสอนจะถูกยุติลงเมื่อค่าความสูญเสียจากผลการทดสอบความถูกต้องหยุดลดลงเป็นเวลาตามจำนวน Epoch ที่กำหนด (กำหนดโดยพารามิเตอร์ `patience` ซึ่งใน YOLO จะมีค่าเริ่มต้นเป็น `100` epochs)

---

## 🔬 กรณีศึกษาการมองเห็นของคอมพิวเตอร์: การวิเคราะห์ Epoch ใน YOLO
ในบันทึกผลการฝึกสอนของคุณ (เช่น ไฟล์ `results.csv` และรูปภาพ `results.png` ที่อยู่ในไดเรกทอรี [runs/detect/train-3](file:///home/luke/ai_training/ultralytics/runs/detect/train-3)):
*   ให้สังเกตค่า `val/box_loss` และ `val/cls_loss`
*   ในการรัน `train-3` การฝึกสอนได้ทำงานเสร็จสิ้นครบ 50 epochs
*   ในช่วงระหว่าง Epoch ที่ 40 ถึง 50 ค่า Validation box loss ยังคงคงที่อยู่ที่ระดับ $\approx 1.38$ ในขณะที่ Training box loss ลดลงจาก $1.49$ เหลือ $1.41$ ซึ่งแสดงว่าจำนวน 50 epochs เป็นระยะเวลาที่เหมาะสมที่สุดแล้ว หากฝึกสอนนานกว่านี้จะเสี่ยงต่อการเกิดการเรียนรู้เกิน (Overfitting)

---

## 💻 โค้ด Python สำหรับพล็อตเส้นโค้งความสามารถในการสรุปนัยทั่วไป

```python
import matplotlib.pyplot as plt

# Mock training logs
epochs = list(range(1, 11))
train_loss = [2.5, 1.8, 1.3, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1]
val_loss   = [2.6, 1.9, 1.5, 1.2, 1.1, 1.1, 1.2, 1.3, 1.5, 1.7]  # Overfits after epoch 6

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
plt.axvline(x=6, color='g', linestyle='--', label='Optimal Stop (Epoch 6)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch Analysis: Overfitting Detection')
plt.legend()
plt.grid(True)
plt.savefig('learning_lab/experiments/02_evaluation_and_tuning/EX30_Epoch_Analysis/epoch_loss_chart.png')
print("Generalization curve chart saved.")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX29_Batch_Size_TH\|EX29: ขนาดกลุ่มข้อมูล (Batch Size)]]
*   [[EX31_Perceptron_TH\|EX31: เพอร์เซปตรอน (The Perceptron)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้ (Learning Journal)]]
