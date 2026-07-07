# 🧠 EX35: โครงข่ายประสาทคอนโวลูชัน (Convolutional Neural Networks - CNN)

โครงข่ายประสาทคอนโวลูชัน (Convolutional Neural Networks หรือ CNN) เป็นสถาปัตยกรรมการเรียนรู้เชิงลึกแบบพิเศษที่ออกแบบมาเพื่อประมวลผลข้อมูลที่มีลักษณะเป็นกริด เช่น ภาพ 2 มิติ โดยอาศัยการแบ่งปันน้ำหนักเชิงพื้นที่ (Spatial Weight Sharing) เพื่อสกัดคุณลักษณะอย่างมีประสิทธิภาพ

---

## 1. ทำไมจึงควรใช้ CNN แทน MLP สำหรับข้อมูลภาพ? (Why CNNs over MLPs for Images?)
หากเราป้อนภาพ RGB ขนาด $640 \times 640$ พิกเซล เข้าไปยังเพอร์เซปตรอนหลายชั้น (MLP) แบบเชื่อมต่อถึงกันอย่างสมบูรณ์:
1.  **การแปลงเป็นเวกเตอร์มิติเดียวทำลายความสัมพันธ์เชิงพื้นที่ (Flattening destroys space):** การแปลงภาพขนาด $640\times640\times3$ ให้เป็นเวกเตอร์ 1 มิติที่มีความยาว $1,228,800$ ค่า จะทำลายความสัมพันธ์เชิงพื้นที่ในระดับท้องถิ่นทั้งหมด (เช่น พิกเซลใดอยู่ข้างพิกเซลใด)
2.  **การระเบิดของพารามิเตอร์ (Parameter Explosion):** หากชั้นซ่อนแรกมีนิวรอน 1,000 ตัว เราจำเป็นต้องใช้พารามิเตอร์:
    $$1,228,800 \times 1,000 = 1.22 \text{ Billion Weights}$$
    ซึ่งนำไปสู่ปัญหาการเรียนรู้เกิน (Overfitting) อย่างรุนแรง และข้อจำกัดด้านหน่วยความจำ

CNN แก้ปัญหานี้โดยใช้ **ขอบเขตการรับรู้เฉพาะที่ (Local Receptive Fields)** (เชื่อมต่อนิวรอนเข้ากับบริเวณพิกเซลเฉพาะที่) และ **การแบ่งปันน้ำหนัก (Weight Sharing)** (ใช้ตัวกรองเดียวกันสไลด์ไปทั่วทั้งภาพ)

---

## 2. สูตรทางคณิตศาสตร์ (Mathematical Formulation)

### A. การดำเนินการคอนโวลูชัน (The Convolution Operation)
ตัวกรอง (Filter หรือ Kernel) คือเมทริกซ์น้ำหนักขนาดเล็ก (เช่น $3 \times 3$) ที่เลื่อนไปบนภาพอินพุต ในแต่ละขั้นตอนจะทำการคูณแบบทีละสมาชิก (Element-wise Multiplication) และหาผลรวมของผลลัพธ์:

$$O(i,j) = (I * K)(i,j) = \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} I(i + m, j + n) \cdot K(m, n)$$

โดยที่ $I$ คือภาพอินพุต, $K$ คือเคอร์เนลขนาด $F \times F$ และ $O$ คือแผนผังคุณลักษณะของเอาต์พุต (Output Feature Map)

### B. สูตรคำนวณรูปร่างเอาต์พุต (Output Shape Formula)
มิติขนาดของแผนผังคุณลักษณะเอาต์พุต (ความกว้าง/ความสูง) ขึ้นอยู่กับขนาดอินพุต ($W$) ขนาดของเคอร์เนล ($F$) ระยะขอบ/การเติมพิกเซล ($P$) และระยะก้าว ($S$):

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - F + 2P}{S} \right\rfloor + 1$$

*   **ระยะก้าว ($S$ - Stride):** ขนาดของขั้นตอนการก้าวเมื่อเลื่อนเคอร์เนล โดยระยะก้าว $2$ จะลดขนาดเอาต์พุตลงครึ่งหนึ่ง (Downsampling)
*   **การเติมพิกเซล ($P$ - Padding):** การเติมค่าศูนย์ที่ขอบภาพอินพุต ช่วยป้องกันไม่ให้มิติเชิงพื้นที่ลดขนาดลงในทุกๆ ชั้น และช่วยรักษาข้อมูลรายละเอียดบริเวณขอบภาพไว้

---

## 💻 ตัวอย่างการทำคอนโวลูชันด้วย PyTorch ในภาษา Python (Python PyTorch Convolution Example)

นี่คือวิธีการรันตัวดำเนินการคอนโวลูชันใน PyTorch และตรวจสอบขนาดเอาต์พุตโดยใช้สูตรข้างต้น:

```python
import torch
import torch.nn as nn

# Create a mock image tensor: Batch=1, Channels=3 (RGB), 640x640 resolution
x = torch.randn(1, 3, 640, 640)

# Define a Convolutional Layer:
# - Input channels = 3
# - Output channels (filters) = 16
# - Kernel size = 3x3
# - Stride = 2
# - Padding = 1
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)

# Run forward pass
output = conv(x)
print(f"Input Shape:  {x.shape}")       # Output: torch.Size([1, 3, 640, 640])
print(f"Output Shape: {output.shape}")  # Output: torch.Size([1, 16, 320, 320])

# Verification using formula:
# W_out = floor((640 - 3 + 2(1)) / 2) + 1
#       = floor(639 / 2) + 1
#       = 319 + 1 = 320
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX34_Backpropagation_TH\|EX34: การแพร่กระจายย้อนกลับ (Backpropagation)]]
*   [[EX36_Feature_Maps_TH\|EX36: แผนผังคุณลักษณะ (Feature Maps)]]
*   กลับไปยังแผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
