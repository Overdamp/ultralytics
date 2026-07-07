# 🧠 EX18: เส้นโค้งความแม่นยำและความระลึก (Precision-Recall Curve)

เส้นโค้งความแม่นยำและความระลึก (Precision-Recall Curve หรือ PR Curve) คือกราฟที่แสดงการชดเชยน้ำหนัก (Trade-off) ระหว่างความแม่นยำ (Precision - ความถูกต้องของผลการทำนาย) และความระลึก (Recall - อัตราการตรวจจับวัตถุได้ครบถ้วน) ของโมเดล ณ ทุกๆ ระดับเกณฑ์ความเชื่อมั่น (Confidence thresholds) ที่เป็นไปได้

---

## 1. ทำไมต้องใช้เส้นโค้ง PR แทนเส้นโค้ง ROC ในงานวิสัยทัศน์คอมพิวเตอร์?

ในการจำแนกประเภทมาตรฐาน เส้นโค้ง ROC (อัตราผลบวกจริง เทียบกับ อัตราผลบวกเท็จ) เป็นที่นิยมอย่างมาก อย่างไรก็ตาม ในงาน **ตรวจจับวัตถุ (Object Detection)** เรามักจะใช้ **เส้นโค้ง PR** เกือบตลอดเวลา 

### ปัญหาของผลลบจริง (True Negative - TN):
*   รูปภาพรูปหนึ่งประกอบด้วยพิกเซลพื้นหลังหรือเซลล์กริด (Grid cells) หลายล้านพิกเซลที่ไม่มีวัตถุใดๆ อยู่เลย 
*   หากเราคำนวณอัตราผลบวกเท็จ (False Positive Rate: FPR):
    $$\text{FPR} = \frac{FP}{TN + FP}$$
    เนื่องจาก $TN$ (ผลลบจริง) มีขนาดใหญ่มหาศาล ตัวหารจึงระเบิดส่งผลให้อัตราผลบวกเท็จ (FPR) มีค่าลู่เข้าใกล้ $0$ เสมอ สิ่งนี้ทำให้เส้นโค้ง ROC ดูดีสมบูรณ์แบบเกินความเป็นจริง ทั้งๆ ที่ในความเป็นจริงแล้วตัวตรวจจับอาจทำนายผิดพลาดอยู่หลายครั้งก็ตาม
*   **ทางออกของเส้นโค้ง PR:** การคำนวณค่าความแม่นยำ (Precision) และความระลึก (Recall) จะ **ไม่** ใช้ผลลบจริง ($TN$) เลย:
    $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
    ทำให้เส้นโค้ง PR มีความไวสูงต่อการประเมินการเตือนลวง ($FP$) และการตรวจจับวัตถุพลาด ($FN$) ส่งผลให้สามารถแสดงประสิทธิภาพที่แท้จริงของตัวตรวจจับวัตถุได้อย่างถูกต้องแม่นยำ

---

## 2. วิธีการตีความเส้นโค้ง PR

*   **ตัวจำแนกประเภทในอุดมคติ (Ideal Classifier):** เส้นโค้งจะโค้งออกไปแตะที่มุมขวาบนพอดี (แสดงถึงความแม่นยำ $100\%$ และความระลึก $100\%$)
*   **การชดเชยน้ำหนัก (The Trade-Off):**
    *   หากคุณตั้งค่าเกณฑ์ความเชื่อมั่นไว้สูง: ความแม่นยำ (Precision) จะสูง แต่ความระลึก (Recall) จะต่ำ (ตรวจพบวัตถุจำนวนน้อยแต่มีความถูกต้องแม่นยำสูงมาก)
    *   หากคุณตั้งค่าเกณฑ์ความเชื่อมั่นไว้ต่ำ: ความระลึก (Recall) จะสูง แต่ความแม่นยำ (Precision) จะต่ำ (ตรวจพบทุกสิ่งทุกอย่างแต่มีผลการแจ้งเตือนลวงผสมมาด้วยจำนวนมาก)
*   **พื้นที่ใต้เส้นโค้ง (Area Under Curve - AUC):** พื้นที่ใต้เส้นโค้ง PR นิยามทางคณิตศาสตร์ว่าคือ **ความแม่นยำเฉลี่ย (Average Precision - AP)** ของคลาสนั้นๆ:

$$\text{AP} = \int_{0}^{1} P(R) \, dR$$

---

## 🔬 ความเชื่อมโยงกับวิสัยทัศน์คอมพิวเตอร์: แผนภูมิ PR ของ YOLO
ในผลการฝึกโมเดลของคุณ (เช่น [runs/detect/train-3/BoxPR_curve.png](file:///home/luke/ai_training/ultralytics/runs/detect/train-3/BoxPR_curve.png)):
*   แผนภูมินี้จะพลอตเส้นโค้ง PR แยกกันสำหรับแต่ละคลาสจากทั้งหมด 26 คลาสของคุณ (เช่น `control-valve` หรือ `flange`)
*   นอกจากนี้ยังแสดงเส้นโค้งหนาสีฟ้าอมเขียว (Cyan) เพื่อเป็นตัวแทนของค่า **mAP@0.5** (Mean Average Precision) ซึ่งเป็นค่าเฉลี่ยของ AP จากทุกคลาสมารวมกัน
*   ยิ่งเส้นโค้งของแต่ละคลาสโค้งนูนเข้าหาขอบบนขวามากเท่าใด ก็หมายความว่าโมเดลตรวจจับวัตถุคลาสนั้นๆ ได้ดีขึ้นเท่านั้น

---

## 💻 การพลอตกราฟเส้นโค้งความแม่นยำและความระลึกด้วย Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Ground truth binary labels (1 = object present, 0 = background)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

# Predicted probabilities (confidence scores) from YOLO
y_scores = np.array([0.9, 0.1, 0.8, 0.75, 0.2, 0.85, 0.4, 0.15, 0.7, 0.3, 0.95, 0.6, 0.05, 0.8, 0.25])

# Calculate Precision-Recall pairs
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

# Plotting the curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'PR Curve (AUC/AP = {pr_auc:.2f})', color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('learning_lab/experiments/02_evaluation_and_tuning/EX18_Precision_Recall_Curve/pr_curve_sample.png')
print("PR Curve plot saved successfully.")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX17_AUC_TH\|EX17: พื้นที่ใต้เส้นโค้ง (AUC)]]
*   [[EX19_Cross_Validation_TH\|EX19: การทดสอบความถูกต้องไขว้ (Cross-Validation)]]
*   กลับไปยังแผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
