# 🧠 EX31: เพอร์เซปตรอน (The Perceptron)

คิดค้นโดย Frank Rosenblatt ในปี ค.ศ. 1958 เพอร์เซปตรอน (Perceptron) เป็นรูปแบบที่ง่ายที่สุดของโครงข่ายประสาทเทียม (Artificial Neural Network) โดยจำลองมาจากโครงสร้างของเซลล์ประสาททางชีววิทยา ทำหน้าที่เป็นตัวจำแนกประเภทสองกลุ่มเชิงเส้น (Linear Binary Classifier)

---

## 1. สูตรทางคณิตศาสตร์ (Mathematical Formulation)

เพอร์เซปตรอนรับอินพุตที่เป็นตัวเลขจำนวน $n$ ค่า นำไปคูณกับน้ำหนัก (Weights) ที่สอดคล้องกัน นำมารวมกับค่าอคติ (Bias) แล้วส่งผลลัพธ์ผ่านฟังก์ชันกระตุ้นการทำงานแบบขั้นบันได (Step Activation Function)

```
[ Inputs: x1, x2, ..., xn ] ---> [ Weighted Sum: z = w*x + b ] ---> [ Heaviside Step Function f(z) ] ---> [ Output: y_hat (0 or 1) ]
```

### A. การรวมกันเชิงเส้น (ผลรวมถ่วงน้ำหนัก)
$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

โดยที่:
*   $\mathbf{x}$ คือเวกเตอร์คุณลักษณะของอินพุต (Input Feature Vector)
*   $\mathbf{w}$ คือเวกเตอร์น้ำหนัก (Weight Vector)
*   $b$ คือค่าอคติ (Bias) (ทำหน้าที่เลื่อนเกณฑ์การเปิดใช้งาน)

### B. ฟังก์ชันกระตุ้นการทำงานแบบขั้นบันไดของเฮวิไซด์ (Heaviside Step Activation Function)
ผลลัพธ์การทำนาย $\hat{y}$ จะถูกกำหนดโดยเกณฑ์ที่เข้มงวด (Hard Threshold):

$$\hat{y} = f(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$

---

## 2. กฎการเรียนรู้ของเพอร์เซปตรอน (The Perceptron Learning Rule)
ในการฝึกสอนเพอร์เซปตรอน น้ำหนักและค่าอคติจะถูกอัปเดตซ้ำๆ ทุกครั้งที่โมเดลเกิดข้อผิดพลาดในการทำนาย

สำหรับแต่ละตัวอย่างข้อมูลฝึกสอน $(\mathbf{x}, y)$:
1.  คำนวณผลการทำนาย $\hat{y}$
2.  อัปเดตน้ำหนักและค่าอคติ:

$$w_i \leftarrow w_i + \eta (y - \hat{y}) x_i$$
$$b \leftarrow b + \eta (y - \hat{y})$$

โดยที่:
*   $y$ คือป้ายกำกับจริงที่เป็นจริง (Ground Truth Label) ($0$ หรือ $1$)
*   $\hat{y}$ คือป้ายกำกับที่ทำนายได้ (Predicted Label) ($0$ หรือ $1$)
*   $\eta$ (Eta) คืออัตราการเรียนรู้ (Learning Rate) ($0 < \eta \le 1$)
*   *หมายเหตุ:* หากทำนายถูกต้อง ($y - \hat{y} = 0$) จะไม่มีการอัปเดตใดๆ เกิดขึ้น

---

## ⚠️ ขีดจำกัดความสามารถในการแยกเชิงเส้น (ปัญหา XOR - The XOR Problem)
เพอร์เซปตรอนตัวเดียวสามารถจำแนกประเภทข้อมูลที่ **สามารถแยกออกได้เชิงเส้น** (Linearly Separable) เท่านั้น (หมายถึงประเภทข้อมูลที่สามารถแบ่งแยกออกจากกันได้ด้วยเส้นตรงเพียงเส้นเดียว)
*   **เกตตรรกะ (Logical Gates):** เพอร์เซปตรอนสามารถแก้ปัญหาเกต `AND`, `OR`, และ `NOT` ได้อย่างง่ายดาย
*   **เกต XOR (The XOR Gate):** เพอร์เซปตรอน **ไม่สามารถ** แก้ปัญหาเกต `XOR` (Exclusive OR) ได้ เนื่องจากประเภทข้อมูลไม่สามารถแบ่งแยกด้วยเส้นตรงได้
*   ข้อจำกัดนี้ (ตีพิมพ์โดย Minsky และ Papert ในปี ค.ศ. 1969) ได้พิสูจน์ให้เห็นว่าเพอร์เซปตรอนชั้นเดียวมีข้อจำกัดอย่างมาก ซึ่งส่งผลให้เกิด **ยุคฤดูหนาวของ AI (AI Winter)** ครั้งแรก จนกระทั่งมีการพัฒนาเพอร์เซปตรอนหลายชั้น (Multi-Layer Perceptrons หรือ MLPs) ที่ใช้ฟังก์ชันกระตุ้นแบบไม่เชิงเส้นขึ้นมา

---

## 💻 การเขียนโปรแกรมเพอร์เซปตรอนในภาษา Python (จากศูนย์)

นี่คือการเขียนโปรแกรมเพอร์เซปตรอนจากศูนย์ (Scratch Implementation) เพื่อแก้ปัญหาเกตตรรกะ `OR`:

```python
import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=0.1):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
        self.lr = lr
        
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if z >= 0 else 0

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error

# Training data for OR Gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

p = Perceptron(input_dim=2)
p.train(X, y, epochs=10)

print("Trained Weights:", p.weights)
print("Trained Bias:", p.bias)
print("Prediction for [1, 0]:", p.predict([1, 0]))  # Outputs 1
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX30_Epoch_Analysis_TH\|EX30: การวิเคราะห์รอบการเรียนรู้ (Epoch Analysis)]]
*   [[EX32_MLP_TH\|EX32: เพอร์เซปตรอนหลายชั้น (Multi-Layer Perceptron - MLP)]]
*   กลับไปยังแผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
