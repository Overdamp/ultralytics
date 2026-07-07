# 🧠 EX27: การจัดตารางอัตราการเรียนรู้ (Learning Rate Scheduling)

อัตราการเรียนรู้ (Learning rate - มักแทนด้วยสัญลักษณ์ $\alpha$ หรือ $\eta$) คือไฮเปอร์พารามิเตอร์ (Hyperparameter) ที่สำคัญที่สุดในการฝึกสอนโครงข่ายประสาทเทียม มันเป็นตัวกำหนดขนาดของก้าว (Step size) ที่จะเคลื่อนที่ไปยังจุดต่ำสุดของฟังก์ชันการสูญเสียในระหว่างการเพิ่มประสิทธิภาพ

---

## 1. ผลกระทบจากขนาดของอัตราการเรียนรู้
*   **อัตราการเรียนรู้สูงเกินไป:** โมเดลจะก้าวเดินในระยะที่ใหญ่เกินไป ส่งผลให้ค่าความสูญเสียแกว่งตัวไปมา หรือแม้กระทั่ง **ลู่ออก (Diverge)** (ค่าน้ำหนักของโมเดลเกิดการระเบิดขึ้น และค่าความสูญเสียกลายเป็น `NaN`)
*   **อัตราการเรียนรู้ต่ำเกินไป:** โมเดลจะก้าวเดินในระยะที่เล็กมาก การฝึกสอนจะล่าช้าเป็นอย่างยิ่ง และมีโอกาสสูงมากที่โมเดลจะติดหล่มอยู่ในค่าต่ำสุดเฉพาะกลุ่มที่ยังไม่ดีพอ (Poor local minima) หรือจุดอานม้า (Saddle points)

เพื่อสร้างสมดุลระหว่างการสำรวจพื้นที่ใหม่ๆ (Exploration) และการใช้ประโยชน์จากพื้นที่เดิม (Exploitation) เราจะใช้ **ตัวจัดตารางอัตราการเรียนรู้ (Learning Rate Schedulers)** เพื่อปรับขนาดของก้าวแบบไดนามิกเมื่อเวลาผ่านไป

---

## 2. กลยุทธ์การจัดตารางอัตราการเรียนรู้ที่พบบ่อย

### A. การลดทอนแบบขั้นบันได (Step Decay)
ลดอัตราการเรียนรู้ด้วยการคูณด้วยปัจจัยลดทอน (เช่น $0.1$) หลังจากครบรอบจำนวน Epoch ที่กำหนด (เช่น ทุกๆ 30 Epoch)

### B. การลดทอนแบบโคไซน์ (Cosine Annealing - ค่าเริ่มต้นของ YOLO)
ปรับลดอัตราการเรียนรู้จากค่าเริ่มต้นสูงสุด (`lr0`) ลงไปยังค่าสุดท้ายต่ำสุด (`lr0 * lrf`) ตามแนวเส้นโค้งโคไซน์ (Cosine curve):

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{cur}}{T_{\max}}\pi\right)\right)$$

วิธีนี้ช่วยให้โมเดลสามารถสำรวจทางเลือกได้กว้างขวางขึ้นในช่วงแรก และลู่เข้าหาจุดต่ำสุดสัมบูรณ์ (Global minimum) อย่างราบรื่นในช่วง Epoch หลังๆ

---

## 3. การอุ่นเครื่องอัตราการเรียนรู้ (Learning Rate Warmup)
ในช่วงเริ่มต้นการฝึกสอน ค่าน้ำหนักจะถูกกำหนดค่าเริ่มต้นแบบสุ่ม การใช้อัตราการเรียนรู้ที่สูงตั้งแต่แรกอาจทำให้เกิดการอัปเดตน้ำหนักที่รุนแรงและไม่สม่ำเสมอ ซึ่งจะทำลายคุณลักษณะสำคัญที่ได้รับการฝึกสอนมาก่อนหน้านี้ (Pre-trained features) เสียหายได้

เพื่อป้องกันปัญหานี้ YOLO จึงใช้ **การอุ่นเครื่องเชิงเส้น (Linear Warmup)**:
*   ในช่วง Epoch แรกๆ (โดยทั่วไปจะตั้งค่า `warmup_epochs=3.0`) อัตราการเรียนรู้จะเริ่มต้นจากค่าใกล้ศูนย์ (กำหนดโดย `warmup_bias_lr` สำหรับพารามิเตอร์ไบแอส) จากนั้นจะค่อยๆ เพิ่มขึ้นเป็นเส้นตรงจนกระทั่งถึงอัตราเป้าหมาย `lr0`

```
Learning Rate (อัตราการเรียนรู้)
  |             / \
  |            /   \
  |           /     \_________________
  |          /                        \
  |_________/__________________________\____
  0       Warmup    Epochs             Total Epochs
```

---

## 💻 ตัวอย่างการจัดตารางอัตราการเรียนรู้ด้วย PyTorch ใน Python

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define a Cosine Annealing Scheduler
# - T_max: Number of epochs to complete the cosine cycle (e.g. 50 epochs)
# - eta_min: Minimum learning rate (final epoch target)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

# Simulated training loop
for epoch in range(50):
    # Train step (dummy loss)
    loss = model(torch.randn(1, 10)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Update learning rate
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Learning Rate = {current_lr:.6f}")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX26_Adam_TH\|EX26: ตัวปรับปรุงประสิทธิภาพแบบ Adam (The Adam Optimizer)]]
*   [[EX28_Weight_Decay_TH\|EX28: การลดทอนน้ำหนัก (Weight Decay)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้ (Learning Journal)]]
