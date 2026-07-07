# 🧠 EX21: การเพิ่มประสิทธิภาพด้วยวิธีการไล่ตามความชัน (Gradient Descent Optimization)

Gradient Descent (การไล่ตามความชัน) คือขั้นตอนวิธี (Algorithm) การเพิ่มประสิทธิภาพแบบวนซ้ำอันดับหนึ่ง (First-order iterative optimization algorithm) ที่ใช้เพื่อลดค่าฟังก์ชันค่าใช้จ่าย (Cost function) $J(\mathbf{w})$ ที่สามารถหาอนุพันธ์ได้ ให้มีค่าต่ำที่สุด โดยการหาค่าต่ำสุดเฉพาะกลุ่ม (Local minimum) หรือค่าต่ำสุดสัมบูรณ์ (Global minimum)

---

## 1. แนวคิดหลัก (Core Concept)
ลองจินตนาการว่าคุณกำลังยืนอยู่บนยอดเขาที่มีหมอกหนา และต้องการหาทางเดินลงไปยังก้นหุบเขา เนื่องจากหมอกบดบังทัศนวิสัย คุณจึงทำได้เพียงตรวจสอบความชันของพื้นดินใต้ฝ่าเท้าของคุณโดยตรงเท่านั้น
*   หากต้องการลงเขาให้เร็วที่สุด คุณต้องก้าวเดินไปในทิศทางที่มี **ความชันลาดลงมากที่สุด**
*   ในการเพิ่มประสิทธิภาพ (Optimization) เวกเตอร์เกรเดียนต์ (Gradient vector) $\nabla J(\mathbf{w})$ จะชี้ไปยังทิศทางของความชันที่ **เพิ่มขึ้น** มากที่สุด (ทางขึ้นเขา)
*   ดังนั้น เพื่อหาค่าต่ำสุด (ก้นหุบเขา) เราจึงต้องเคลื่อนที่ไปใน **ทิศทางตรงกันข้าม** (โดยการลบเกรเดียนต์ออก)

---

## 2. สูตรทางคณิตศาสตร์ (Mathematical Formulation)

สำหรับเวกเตอร์น้ำหนัก (Weight vector) $\mathbf{w}$ กฎการอัปเดตคือ:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla J(\mathbf{w}_t)$$

โดยที่:
*   $\mathbf{w}_t$ คือเวกเตอร์น้ำหนักในปัจจุบัน
*   $\mathbf{w}_{t+1}$ คือเวกเตอร์น้ำหนักที่อัปเดตแล้ว
*   $\alpha > 0$ คือ **อัตราการเรียนรู้ (Learning Rate)** หรือขนาดของก้าว (Step size)
*   $\nabla J(\mathbf{w}_t)$ คือ **เวกเตอร์เกรเดียนต์ (Gradient vector)** ของอนุพันธ์ย่อย (Partial derivatives):

$$\nabla J(\mathbf{w}) = \left[ \frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2}, \dots, \frac{\partial J}{\partial w_n} \right]^T$$

---

## 3. พื้นผิวฟังก์ชันการสูญเสียแบบคอนเวกซ์และไม่ใช่คอนเวกซ์ (Convex vs. Non-Convex Loss Landscapes)
*   **ฟังก์ชันการสูญเสียแบบคอนเวกซ์ (Convex Loss Function - เช่น Linear Regression OLS):** พื้นผิวของการสูญเสียจะมีลักษณะคล้ายชาม (มีค่าต่ำสุดสัมบูรณ์เพียงจุดเดียว) วิธีการไล่ตามความชันได้รับการรับรองทางคณิตศาสตร์ว่าจะสามารถค้นพบค่าต่ำสุดสัมบูรณ์ (Global minimum) ได้อย่างแน่นอน หากตั้งค่าอัตราการเรียนรู้ (Learning Rate) ได้อย่างเหมาะสม
*   **ฟังก์ชันการสูญเสียแบบไม่ใช่คอนเวกซ์ (Non-Convex Loss Function - เช่น โครงข่ายประสาทเทียมเชิงลึกอย่าง YOLO):** พื้นผิวของการสูญเสียมีความซับซ้อนสูง ประกอบไปด้วยค่าต่ำสุดเฉพาะกลุ่ม (Local minima) หลายจุด พื้นราบ (Flat plateaus) และ **จุดอานม้า (Saddle points)** (จุดที่ความชันเป็นศูนย์แต่ไม่ใช่จุดต่ำสุด) วิธีการไล่ตามความชันแบบพื้นฐานอาจติดหล่มอยู่ในพื้นที่เหล่านี้ได้ง่าย ซึ่งจำเป็นต้องใช้ตัวปรับปรุงประสิทธิภาพแบบปรับตัวได้ (Adaptive optimizers เช่น Adam) หรือการอัปเดตแบบสุ่ม (Stochastic updates เช่น SGD) เพื่อหลุดพ้นจากอุปสรรคเหล่านี้

---

## 💻 การเขียนโปรแกรม Gradient Descent ด้วย Python

นี่คือตัวอย่างการเขียนโค้ด Gradient Descent แบบเริ่มต้นใหม่ทั้งหมด (Scratch implementation) เพื่อหาค่าต่ำสุดของฟังก์ชันกำลังสอง $f(x) = x^2$ (โดยค่าต่ำสุดจากการคำนวณทางคณิตศาสตร์จะอยู่ที่ $x=0$):

```python
# Function: f(x) = x^2
def cost_function(x):
    return x**2

# Derivative: df/dx = 2 * x
def compute_gradient(x):
    return 2 * x

# Parameters
x = 10.0  # Starting point
lr = 0.1  # Learning rate
epochs = 20

print("Starting Gradient Descent:")
for epoch in range(epochs):
    # Compute gradient (slope)
    gradient = compute_gradient(x)
    
    # Update position (move opposite to slope)
    x = x - lr * gradient
    
    cost = cost_function(x)
    print(f"Epoch {epoch+1}: x = {x:.4f}, Cost = {cost:.4f}")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX20_Bias_Variance_TH\|EX20: ความเอนเอียงและความแปรปรวน (Bias-Variance Tradeoff)]]
*   [[EX22_Stochastic_GD_TH\|EX22: การไล่ตามความชันแบบสุ่ม (Stochastic Gradient Descent)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้ (Learning Journal)]]
