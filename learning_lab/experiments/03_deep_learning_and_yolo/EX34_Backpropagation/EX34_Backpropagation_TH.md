# 🧠 EX34: การแพร่กระจายย้อนกลับและกฎลูกโซ่ (Backpropagation & The Chain Rule)

การแพร่กระจายย้อนกลับ (Backpropagation หรือ Backward Propagation of Errors) เป็นอัลกอริทึมหลักที่ใช้ในการฝึกสอนโครงข่ายประสาทเทียม ทำหน้าที่คำนวณค่าความชัน (Gradient) ของฟังก์ชันการสูญเสีย (Loss Function) เทียบกับน้ำหนัก (Weight) และค่าอคติ (Bias) แต่ละตัว ซึ่งช่วยให้ตัวเพิ่มประสิทธิภาพ (Optimizer) สามารถอัปเดตพารามิเตอร์ต่างๆ และลดข้อผิดพลาดให้เหลือน้อยที่สุดได้

---

## 1. วัฏจักรการเรียนรู้แบบสองขั้นตอน (The Two-Pass Learning Cycle)
1.  **ขั้นตอนการส่งผ่านไปข้างหน้า (Forward Pass):** คุณลักษณะอินพุต (Input Features) จะส่งผ่านไปข้างหน้าตามชั้นต่างๆ ของโครงข่าย โดยจะมีการคำนวณการรวมกันเชิงเส้นและส่งผ่านฟังก์ชันกระตุ้นการทำงาน เพื่อผลิตผลการทำนายและค่าการสูญเสียสุดท้าย (Loss value, $L$)
2.  **ขั้นตอนการแพร่กระจายย้อนกลับ (Backward Pass):** เริ่มต้นที่ชั้นเอาต์พุต อัลกอริทึมจะคำนวณความชันของ Loss เทียบกับน้ำหนักและค่าอคติทุกตัว แล้วส่งผ่านค่าอนุพันธ์เหล่านี้ย้อนกลับไปโดยใช้ **กฎลูกโซ่ (Chain Rule)**

---

## 2. การอนุพัทธ์ทางคณิตศาสตร์ (ตัวอย่างนิวรอนเดี่ยว)

ลองมาพิจารณานิวรอนที่มีอินพุตเดียวแบบง่ายๆ:

```
[ Input: x ] ---> ( Linear Combine: z = w*x + b ) ---> ( Activation: a = f(z) ) ---> [ Prediction: a ] ---> ( Loss: L = (a - y)^2 )
```

เป้าหมายของเราคือการคำนวณหา $\frac{\partial L}{\partial w}$ (การเปลี่ยนแปลงของน้ำหนัก $w$ ส่งผลต่อค่า Loss $L$ อย่างไร) เพื่อนำไปอัปเดตค่า $w$

โดยใช้กฎลูกโซ่ (Chain Rule):

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### การอนุพัทธ์ทีละขั้นตอน:
1.  **ความชันของ Loss ($\frac{\partial L}{\partial a}$):** อัตราการเปลี่ยนแปลงของ Loss เทียบกับผลการทำนาย $a$:
    $$\frac{\partial L}{\partial a} = \frac{\partial}{\partial a}(a - y)^2 = 2(a - y)$$
2.  **ความชันของฟังก์ชันกระตุ้น ($\frac{\partial a}{\partial z}$):** อัตราการเปลี่ยนแปลงของผลการทำนายเทียบกับผลรวมก่อนการกระตุ้น $z$:
    $$\frac{\partial a}{\partial z} = \frac{\partial}{\partial z}f(z) = f'(z)$$
3.  **ความชันเชิงเส้น ($\frac{\partial z}{\partial w}$):** อัตราการเปลี่ยนแปลงของผลรวมเทียบกับน้ำหนัก $w$:
    $$\frac{\partial z}{\partial w} = \frac{\partial}{\partial w}(w \cdot x + b) = x$$

### ค่าความชันสุดท้าย (Final Gradient):
การรวมสามส่วนประกอบนี้เข้าด้วยกันจะได้:

$$\frac{\partial L}{\partial w} = 2(a - y) \cdot f'(z) \cdot x$$

---

## ⚠️ ความท้าทายในโครงข่ายประสาทที่มีความลึกมาก: ความชันสูญหายและความชันระเบิด (Challenges in Deep Networks: Vanishing & Exploding Gradients)
ในโครงข่ายที่มีความลึกมาก (เช่น YOLO ซึ่งอาจมีมากกว่า 200 ชั้น) กฎลูกโซ่จะทำการคูณค่าอนุพันธ์ต่อเนื่องกันไปในทุกๆ ชั้น:
*   **ความชันสูญหาย (Vanishing Gradients):** หากอนุพันธ์ของฟังก์ชันกระตุ้นมีค่าน้อย (เช่น ค่าอนุพันธ์สูงสุดของ Sigmoid คือ $0.25$) การคูณซ้ำๆ กันหลายครั้งจะทำให้ค่าความชันลดลงเรื่อยๆ จนเหลือ 0 ในชั้นแรกๆ ส่งผลให้ชั้นเหล่านั้นหยุดเรียนรู้
*   **ความชันระเบิด (Exploding Gradients):** หากค่าอนุพันธ์มีค่ามาก ($> 1$) การคูณซ้ำๆ กันจะทำให้ความชันเติบโตขึ้นแบบทวีคูณ (Exponential) ส่งผลให้น้ำหนักไร้เสถียรภาพ
*   **แนวทางการแก้ไขใน YOLO:** YOLO ใช้ฟังก์ชันกระตุ้นแบบ SiLU (Swish) เพื่อป้องกันการสูญหายของความชัน และใช้ **การเชื่อมต่อแบบข้ามชั้น/ส่วนที่เหลือ (Skip/Residual Connections)** (เช่น โมดูล C3k2/C2PSA) ซึ่งช่วยให้ค่าความชันสามารถไหลย้อนกลับได้โดยตรงโดยไม่หดตัวลง

---

## 💻 การเขียนโปรแกรมสาธิตระบบหาอนุพันธ์อัตโนมัติ (Autograd) ของ PyTorch ในภาษา Python (Python PyTorch Autograd Demonstration)

PyTorch ช่วยให้การแพร่กระจายย้อนกลับทำงานโดยอัตโนมัติด้วยกลไก `autograd`:

```python
import torch

# Initialize weight, input, and target with gradient tracking enabled
x = torch.tensor([1.5])
w = torch.tensor([0.8], requires_grad=True)
b = torch.tensor([0.2], requires_grad=True)
y = torch.tensor([2.0])  # Target

# 1. Forward Pass
z = w * x + b
a = torch.sigmoid(z)  # Sigmoid activation
loss = (a - y) ** 2   # Squared error loss

# 2. Backward Pass
loss.backward()  # Calculates gradients automatically

# Output computed gradients: dL/dw and dL/db
print(f"Computed Gradient dL/dw: {w.grad.item():.4f}")
print(f"Computed Gradient dL/db: {b.grad.item():.4f}")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX33_Activation_Function_TH\|EX33: ฟังก์ชันกระตุ้นการทำงาน (Activation Functions)]]
*   [[EX35_CNN_TH\|EX35: โครงข่ายประสาทคอนโวลูชัน (Convolutional Neural Networks - CNN)]]
*   กลับไปยังแผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
