# 🧠 EX16: เส้นโค้ง ROC (ROC Curve)

เส้นโค้งลักษณะเฉพาะในการทำงานของตัวรับ (Receiver Operating Characteristic หรือ ROC Curve) คือกราฟที่แสดงความสามารถในการจำแนกประเภทของระบบตัวจำแนกประเภทแบบสองกลุ่ม (Binary classifier) เมื่อปรับเปลี่ยนเกณฑ์การแบ่งประเภท (Discrimination threshold) ไปในระดับต่างๆ

---

## 1. ประวัติและความเป็นมา
เส้นโค้ง ROC ถูกพัฒนาขึ้นครั้งแรกในช่วงสงครามโลกครั้งที่สองโดยวิศวกรเรดาร์ (เพื่อแยกแยะสัญญาณของเครื่องบินญี่ปุ่นออกจากสัญญาณรบกวนของเรดาร์) ในปัจจุบัน เส้นโค้ง ROC ได้กลายมาเป็นเครื่องมือมาตรฐานสำหรับประเมินว่าโมเดลสามารถแยกแยะระหว่างคลาสบวกและคลาสลบได้ดีเพียงใด

---

## 2. พิกัดทางคณิตศาสตร์

เส้นโค้ง ROC จะพลอตโดยมี:
*   **แกน Y:** อัตราผลบวกจริง (True Positive Rate: TPR) ซึ่งเป็นที่รู้จักกันในชื่อ **ความระลึก (Recall)** หรือ **ความไว (Sensitivity)**:
    $$\text{TPR} = \frac{TP}{TP + FN}$$
*   **แกน X:** อัตราผลบวกเท็จ (False Positive Rate: FPR) หรือที่รู้จักกันในชื่อ ความน่าจะเป็นของการเตือนลวง โดยทางคณิตศาสตร์จะมีค่าเท่ากับ $1 - \text{Specificity}$:
    $$\text{FPR} = \frac{FP}{TN + FP} = 1 - \frac{TN}{TN + FP}$$

---

## 3. วิธีการตีความเส้นโค้ง ROC

เส้นโค้งนี้สร้างขึ้นจากการคำนวณค่า TPR และ FPR ณ ทุกระดับเกณฑ์การจำแนกประเภทที่เป็นไปได้ (ตั้งแต่ $1.0$ ไล่ลงมาถึง $0.0$):

```
TPR (Recall)
  ^
  |       จุดสมบูรณ์แบบ (0,1)
  |      *-----------
  |     /           |   <-- ตัวจำแนกประเภทที่ดีเยี่ยม (โค้งเข้าหาบนซ้าย)
  |    /            |
  |   /   /         |   <-- เส้นทแยงมุมเดาสุ่ม (y = x)
  |  /  /           |
  |/_/______________|____> FPR
  0                 1.0
```

1.  **เส้นทแยงมุม ($y = x$):** แสดงถึงการเดาสุ่ม (เหมือนการโยนเหรียญหัวก้อย) พื้นที่ใต้เส้นทแยงมุมนี้มีค่าเท่ากับ $0.5$ พอดี
2.  **มุมบนซ้าย $(0, 1)$:** แสดงถึงโมเดลในอุดมคติ (มีความระลึก $100\%$ และไม่มีการเตือนลวงเลย)
3.  **การเคลื่อนไหวตามเกณฑ์ (Threshold):**
    *   ที่เกณฑ์ $= 1.0$: โมเดลจะไม่ทำนายข้อมูลใดเป็นคลาสบวกเลย พิกัดจะอยู่ที่ $(0,0)$
    *   ที่เกณฑ์ $= 0.0$: โมเดลจะทำนายข้อมูลทั้งหมดเป็นคลาสบวก พิกัดจะอยู่ที่ $(1,1)$
    *   เมื่อเราลดเกณฑ์ลงจาก $1.0$ ไปเป็น $0.0$ จุดพิกัดจะเลื่อนไปตามเส้นโค้งจากมุมล่างซ้ายขึ้นไปหามุมบนขวา

---

## 💻 การพลอตกราฟเส้นโค้ง ROC ด้วย Python

นี่คือสคริปต์ Python ที่ใช้ Scikit-Learn ในการคำนวณเกณฑ์ ROC และพลอตกราฟเส้นโค้ง:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Ground truth binary labels (1 = positive class, 0 = negative class)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

# Raw prediction scores (probabilities output by the model)
y_scores = np.array([0.9, 0.1, 0.8, 0.75, 0.2, 0.85, 0.4, 0.15, 0.7, 0.3, 0.95, 0.6, 0.05, 0.8, 0.25])

# Calculate ROC curve coordinates
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('learning_lab/experiments/02_evaluation_and_tuning/EX16_ROC_Curve/roc_curve_sample.png')
print("ROC Curve plot saved successfully.")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX15_F1_Score_TH\|EX15: คะแนน F1 (F1 Score)]]
*   [[EX17_AUC_TH\|EX17: พื้นที่ใต้เส้นโค้ง (AUC)]]
*   กลับไปยังแผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
