# 🧠 EX32: เพอร์เซปตรอนหลายชั้น (Multi-Layer Perceptron - MLP)

เพอร์เซปตรอนหลายชั้น (Multi-Layer Perceptron หรือ MLP) เป็นประเภทหนึ่งของโครงข่ายประสาทเทียมแบบป้อนไปข้างหน้า (Feedforward Artificial Neural Network - ANN) ซึ่งประกอบด้วยชั้นอินพุต (Input Layer) ชั้นซ่อน (Hidden Layer) ของนิวรอนอย่างน้อยหนึ่งชั้นขึ้นไป และชั้นเอาต์พุต (Output Layer)

---

## 1. สถาปัตยกรรมของ MLP (MLP Architecture)
MLP เป็นแบบเชื่อมต่อถึงกันอย่างสมบูรณ์ (Fully Connected หรือ Dense) ซึ่งหมายความว่านิวรอนทุกตัวในชั้น $l$ จะเชื่อมต่อกับนิวรอนทุกตัวในชั้นก่อนหน้า $l-1$:

```
[ Input Vector x ] ---> [ Hidden Layer (a_1) ] ---> [ Hidden Layer (a_2) ] ---> [ Output Layer y_hat ]
```

---

## 2. สมการการแพร่กระจายไปข้างหน้าแบบเวกเตอร์ (Vectorized Forward Propagation Equations)

ในการคำนวณผลการทำนายในโครงข่ายประสาทเทียม เราจะแสดงคณิตศาสตร์ในรูปแบบเมทริกซ์ที่แปลงเป็นเวกเตอร์ (Vectorized Matrix Form) สำหรับโครงข่ายที่มีชั้นซ่อนหนึ่งชั้นดังนี้:

### ชั้นที่ 1 (ชั้นซ่อน - Hidden Layer):
1.  **การรวมกันเชิงเส้น (Linear Combination):** คูณเวกเตอร์อินพุต $\mathbf{x}$ ด้วยเมทริกซ์น้ำหนัก $\mathbf{W}^{[1]}$ และบวกด้วยเวกเตอร์อคติ $\mathbf{b}^{[1]}$:
    $$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$$
2.  **เอาต์พุตจากฟังก์ชันกระตุ้น (Activation Output):** ส่งผลรวมผ่านฟังก์ชันกระตุ้นแบบไม่เชิงเส้น $g^{[1]}$:
    $$\mathbf{a}^{[1]} = g^{[1]}(\mathbf{z}^{[1]})$$

### ชั้นที่ 2 (ชั้นเอาต์พุต - Output Layer):
1.  **การรวมกันเชิงเส้น (Linear Combination):** ใช้เอาต์พุตของชั้นซ่อน $\mathbf{a}^{[1]}$ เป็นอินพุตของชั้นถัดไป:
    $$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$
2.  **การทำนายสุดท้าย ($\mathbf{\hat{y}}$):** ส่งผ่านฟังก์ชันกระตุ้นของเอาต์พุต $g^{[2]}$ (เช่น Softmax สำหรับการจำแนกประเภท หรือ Linear สำหรับการถดถอย):
    $$\mathbf{\hat{y}} = \mathbf{a}^{[2]} = g^{[2]}(\mathbf{z}^{[2]})$$

---

## 💡 ทฤษฎีการประมาณค่าสากล (The Universal Approximation Theorem)
ทฤษฎีนี้ระบุว่า โครงข่ายประสาทแบบป้อนไปข้างหน้าที่มี **ชั้นซ่อนเพียงชั้นเดียว** และมีจำนวนนิวรอนจำกัด สามารถประมาณค่าฟังก์ชันต่อเนื่องใดๆ ก็ได้ หากฟังก์ชันกระตุ้นการทำงานเป็นแบบไม่เชิงเส้น
*   **พลังของความไม่เชิงเส้น (The Power of Non-Linearity):** หากเราถอดฟังก์ชันกระตุ้นการทำงานออก ชั้นเชิงเส้นหลายชั้นจะยุบรวมกันทางคณิตศาสตร์กลายเป็นชั้นเชิงเส้นเพียงชั้นเดียว ซึ่งจำกัดให้โครงข่ายสามารถจำลองได้เฉพาะเส้นตรงแบบง่ายๆ เท่านั้น ฟังก์ชันกระตุ้นแบบไม่เชิงเส้นช่วยให้ MLP สามารถเรียนรู้เส้นแบ่งเขตที่มีความโค้งและซับซ้อนได้

---

## 💻 การเขียนโปรแกรม MLP ด้วย PyTorch ในภาษา Python (Python PyTorch MLP Implementation)

```python
import torch
import torch.nn as nn

# Define an MLP Model:
# - Input size = 4 features
# - Hidden layer = 8 neurons (with ReLU activation)
# - Output layer = 2 classes (logits)
mlp = nn.Sequential(
    nn.Linear(in_features=4, out_features=8),
    nn.ReLU(),
    nn.Linear(in_features=8, out_features=2)
)

# Mock input data (Batch size = 1, Features = 4)
x = torch.tensor([[1.5, -0.5, 2.0, 0.0]])

# Forward pass
output = mlp(x)
print(f"Input Tensor:  {x}")
print(f"Output Logits: {output}")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX31_Perceptron_TH\|EX31: เพอร์เซปตรอน (The Perceptron)]]
*   [[EX33_Activation_Function_TH\|EX33: ฟังก์ชันกระตุ้นการทำงาน (Activation Functions)]]
*   กลับไปยังแผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
