# 🧠 EX28: การเพิ่มประสิทธิภาพด้วยการลดทอนน้ำหนัก (Weight Decay Optimization)

การลดทอนน้ำหนัก (Weight decay) คือเทคนิคการทำให้เป็นระเบียบ (Regularization) ที่ช่วยป้องกันการเรียนรู้เกิน (Overfitting) โดยการเพิ่มค่าปรับ (Penalty) ที่แปรผันตามขนาดของค่าน้ำหนักในโครงข่ายประสาทเทียม ส่งผลให้ค่าน้ำหนักลดทอนลงเข้าใกล้ศูนย์

---

## 1. สูตรทางคณิตศาสตร์ (Mathematical Formulation)

ใน Stochastic Gradient Descent (SGD) แบบมาตรฐาน การลดทอนน้ำหนักจะมีผลลัพธ์ทางคณิตศาสตร์เทียบเท่ากับ **L2 Regularization**

### A. ฟังก์ชันค่าใช้จ่ายของ L2 Regularization:
เราเพิ่มพจน์ค่าปรับ (Penalty term) เข้าไปในฟังก์ชันการสูญเสียหลัก $E_0(\mathbf{w})$:

$$E(\mathbf{w}) = E_0(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

โดยที่ $\lambda$ คือสัมประสิทธิ์การลดทอนน้ำหนัก (Weight decay coefficient)

### B. การหาอนุพันธ์สำหรับการอัปเดตเกรเดียนต์:
การคำนวณเกรเดียนต์ของฟังก์ชันการสูญเสียเทียบกับน้ำหนัก:

$$\nabla E(\mathbf{w}) = \nabla E_0(\mathbf{w}) + \lambda \mathbf{w}$$

ขั้นตอนการอัปเดตพารามิเตอร์คือ:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla E(\mathbf{w}_t) = \mathbf{w}_t - \alpha \left( \nabla E_0(\mathbf{w}_t) + \lambda \mathbf{w}_t \right)$$

การจัดรูปสมการใหม่:

$$\mathbf{w}_{t+1} = \mathbf{w}_t(1 - \alpha \lambda) - \alpha \nabla E_0(\mathbf{w}_t)$$

สิ่งนี้พิสูจน์ให้เห็นว่า ในแต่ละ Epoch น้ำหนักจะถูกคูณด้วยปัจจัยการหดตัว (Shrinkage factor) $(1 - \alpha \lambda)$ (ซึ่งมีค่าน้อยกว่า 1 เล็กน้อย) ก่อนที่จะนำไปลบออกด้วยเกรเดียนต์ของค่าสูญเสียตั้งต้น

---

## 2. การลดทอนน้ำหนักแบบแยกออก (Decoupled Weight Decay - AdamW)
แม้ว่า L2 regularization และ Weight Decay จะเหมือนกันทุกประการใน SGD แต่พวกมันจะ **แยกออกจากกันในตัวปรับปรุงประสิทธิภาพแบบปรับตัวได้ (Adaptive optimizers)** เช่น Adam

ใน Adam แบบมาตรฐาน การเพิ่ม L2 regularization ลงในค่าสูญเสียจะส่งผ่านพจน์ของการลดทอนน้ำหนักไปยังตัวประมาณค่าเฉลี่ยเคลื่อนที่ของเกรเดียนต์ (Moving average gradient estimators) สิ่งนี้ทำให้น้ำหนักที่มีเกรเดียนต์ในอดีตขนาดใหญ่ลดทอนลง *น้อยกว่า* น้ำหนักที่มีเกรเดียนต์ขนาดเล็ก ซึ่งเป็นผลลัพธ์ที่ตรงกันข้ามกับสิ่งที่ควรจะเป็น

**AdamW** แก้ไขปัญหานี้โดยการแยกขั้นตอนการลดทอนน้ำหนักออกมา และใช้ปรับกับค่าพารามิเตอร์โดยตรง แทนที่จะเพิ่มเข้าไปในการคำนวณเกรเดียนต์:

$$\mathbf{w}_{t+1} = \mathbf{w}_t(1 - \alpha \lambda) - \text{Adam\_Update}(\nabla E_0(\mathbf{w}_t))$$

เนื่องจาก YOLO ใช้ AdamW เป็นค่าเริ่มต้น แนวทางการแยกการอัปเดตนี้จึงมีความสำคัญอย่างยิ่งต่อการรักษาความเสถียรในการฝึกสอน

---

## 💻 การใช้งานตัวปรับปรุงประสิทธิภาพใน PyTorch ด้วย Python

นี่คือวิธีการกำหนดค่าวัดการลดทอนน้ำหนัก (Weight decay) ในตัวปรับปรุงประสิทธิภาพของ PyTorch:

```python
# Function: f(x) = x^2 (จากตัวอย่างการฝึกสอนแบบจำลอง)
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10, 2))

# Scenario A: SGD with Weight Decay (Equivalent to L2)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005)

# Scenario B: AdamW (Decoupled Weight Decay - YOLO Default)
# This keeps weight decay separate from the adaptive gradient moment estimators.
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX27_Learning_Rate_TH\|EX27: การจัดตารางอัตราการเรียนรู้ (Learning Rate Scheduling)]]
*   [[EX29_Batch_Size_TH\|EX29: การวิเคราะห์ขนาดกลุ่มข้อมูล (Batch Size Analysis)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้ (Learning Journal)]]
