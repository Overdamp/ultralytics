# 🧠 EX71: การติดตามสำหรับการวิเคราะห์การกีฬา (Kalman Filters สำหรับวัตถุเคลื่อนที่เร็ว)

ในการวิเคราะห์ข้อมูลการกีฬา การติดตามวัตถุขนาดเล็กที่เคลื่อนที่ด้วยความเร็วสูง (เช่น ลูกเทนนิส ลูกกอล์ฟ หรือลูกเบสบอล) ถือเป็นหนึ่งในงานที่ท้าทายที่สุดในด้านคอมพิวเตอร์วิทัศน์ ตัวตรวจจับวัตถุมาตรฐานอย่าง YOLO มักจะประสบปัญหาดังนี้:
*   **ความเบลอจากการเคลื่อนที่ (Motion Blur):** วัตถุถูกยืดออกหรือเบลอข้ามพิกเซลเนื่องจากความเร็วที่สูงมากและอัตราความเร็วชัตเตอร์ของกล้องต่ำ ทำให้ตรวจจับผิดพลาดหรือตรวจจับไม่พบเลย
*   **การถูกบดบังอย่างรุนแรง (Severe Occlusion):** วัตถุถูกบังโดยผู้เล่น อุปกรณ์กีฬา (ไม้แร็กเก็ต ตาข่าย) หรือฉากหลังที่สับสนปนเป

เพื่อเอาชนะความล้มเหลวเหล่านี้ เราจึงนำการตรวจจับของ YOLO มารวมกับ **ตัวกรองคัลมาน (Kalman Filter)** ตัวกรองคัลมานเป็นตัวประมาณสถานะแบบเวียนเกิด (Recursive state estimator) ที่ใช้การสร้างแบบจำลองทางฟิสิกส์ของการเคลื่อนที่ของวัตถุ (เช่น ความเร็วและตำแหน่ง) เพื่อทำนายตำแหน่งของวัตถุในเฟรมที่ YOLO ไม่สามารถตรวจจับได้

---

## 1. หลักการสำคัญและสูตรทางคณิตศาสตร์

สถานะของลูกบอลที่กำลังเคลื่อนที่ในปริภูมิ 2 มิติ สามารถสร้างแบบจำลองได้จากตำแหน่ง ($x, y$) และความเร็ว ($v_x, v_y$) ของมัน เราสามารถกำหนดเวกเตอร์สถานะ $\mathbf{x}_k$ ณ ขั้นเวลา (Time step) $k$ ได้ดังนี้:

$$\mathbf{x}_k = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}_k$$

ตัวกรองคัลมานทำงานโดยสลับกันระหว่าง 2 เฟสหลัก ได้แก่: **การทำนาย (Predict)** และ **การปรับปรุงสถานะ (Update)**

### ขั้นตอนที่ 1: การทำนาย (Motion Model)
เมื่อใช้หลักฟิสิกส์ (สมมติให้ความเร็วคงที่ในช่วงเวลาสั้นๆ $\Delta t$) เราจะทำนายสถานะ $\mathbf{x}_{k|k-1}$ และเมทริกซ์ความแปรปรวนร่วมของสถานะ (State covariance matrix) $\mathbf{P}_{k|k-1}$:

$$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1}$$

$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$$

โดยที่:
*   $\mathbf{F}$ คือเมทริกซ์การเปลี่ยนผ่านสถานะ (State transition matrix):
    $$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
*   $\mathbf{Q}$ คือเมทริกซ์ความแปรปรวนร่วมของสัญญาณรบกวนในกระบวนการ (Process noise covariance matrix) ซึ่งแสดงถึงความไม่แน่นอนในแบบจำลองทางฟิสิกส์ (เช่น แรงต้านอากาศ ลม หรือการเบี่ยงเบนของแรงโน้มถ่วง)

### ขั้นตอนที่ 2: การปรับปรุงสถานะ (Sensor Measurement)
เมื่อ YOLO ตรวจจับลูกบอลได้สำเร็จ จะได้เวกเตอร์การวัดค่า $\mathbf{z}_k = [x_{yolo}, y_{yolo}]^T$ เราจะปรับปรุงสถานะที่ทำนายไว้โดยใช้เมทริกซ์การวัดค่า $\mathbf{H}$ และอัตราขยายคัลมาน (Kalman Gain) $\mathbf{K}_k$:

$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R})^{-1}$$

$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1})$$

$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$$

โดยที่:
*   $\mathbf{H}$ คือเมทริกซ์การวัดค่าที่แปลงข้อมูลสถานะไปเป็นมิติของการวัดค่าจริง:
    $$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
*   $\mathbf{R}$ คือเมทริกซ์ความแปรปรวนร่วมของสัญญาณรบกวนจากการวัด (Measurement noise covariance matrix) ซึ่งสะท้อนถึงความแปรปรวนของพิกัดกรอบล้อมรอบจาก YOLO

**การจัดการเมื่อเกิดการบดบังหรือภาพเบลอ (Handling Occlusions/Blur):** หาก YOLO ไม่สามารถตรวจจับลูกบอลได้ในเฟรม $k$ เราจะข้ามขั้นตอน **การปรับปรุงสถานะ (Update)** ไปเลย โดยสถานะสุดท้ายที่ประมาณการได้จะเป็นเพียงค่าสถานะจากการทำนายเท่านั้น: $\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1}$ และ $\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1}$

---

## 💻 การเขียนโค้ดด้วย Python

ด้านล่างนี้คือโค้ดตัวอย่างการคำนวณตัวกรองคัลมานแบบ 2 มิติเพื่อติดตามทิศทางของลูกบอลที่เคลื่อนที่ตามแนวเส้นตรง โดยจำลองเหตุการณ์ที่ตัวตรวจจับ YOLO ตรวจหาลูกบอลไม่เจอในเฟรมกลางๆ เนื่องจากถูกบัง

```python
import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=0.1, measurement_noise=2.0):
        # State: [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        
        # State Transition Matrix (F)
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])
        
        # Measurement Matrix (H)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance Matrices
        self.P = np.eye(4) * 10.0  # Initial uncertainty
        self.Q = np.eye(4) * process_noise  # Process noise
        self.R = np.eye(2) * measurement_noise  # Measurement noise
        
    def predict(self):
        # Predict State and Covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2].flatten()
        
    def update(self, z):
        # Measurement update (z is [x, y] from YOLO)
        z = np.array(z).reshape(2, 1)
        y = z - np.dot(self.H, self.x)  # Innovation/Residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman Gain
        
        # Update State and Covariance
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)
        return self.x[:2].flatten()

# Simulation Setup
frames = 10
actual_velocity = (10, 5)  # pixels per frame (vx, vy)
kf = KalmanFilter2D(dt=1.0)

# Initialize state with first frame detection
kf.x = np.array([[10], [20], [10], [5]])  # Initial pos (10, 20), vel (10, 5)

# Simulate 10 frames
# YOLO detections exist for all frames EXCEPT frames 4, 5, and 6 (occluded)
yolo_detections = {
    0: [10, 20],
    1: [21, 24],
    2: [29, 31],
    3: [41, 35],
    4: None,  # Occluded!
    5: None,  # Occluded!
    6: None,  # Occluded!
    7: [80, 56],
    8: [91, 61],
    9: [102, 64]
}

print("Frame | Ground Truth | YOLO Detection | Kalman Estimate | Status")
print("-" * 75)

for f in range(frames):
    # Compute ground truth position
    gt_x = 10 + f * actual_velocity[0]
    gt_y = 20 + f * actual_velocity[1]
    
    # Step 1: Predict next position
    predicted_pos = kf.predict()
    
    detection = yolo_detections[f]
    
    if detection is not None:
        # Step 2: Update Kalman filter if YOLO detected the ball
        final_estimate = kf.update(detection)
        status = "Updated"
    else:
        # No detection: Rely purely on Kalman Prediction
        final_estimate = predicted_pos
        status = "Predicted (Occluded)"
        
    print(f"{f:5d} | ({gt_x:3d}, {gt_y:3d})    | "
          f"{str(detection):14s} | ({final_estimate[0]:.1f}, {final_estimate[1]:.1f}) | {status}")
```

---

## 🛠️ ความเชื่อมโยงกับคอมพิวเตอร์วิทัศน์และ YOLO (Connection to CV & YOLO)

*   **โมเดลปริภูมิสถานะในตัวติดตามวัตถุ (State Space Models in Trackers):** ตัวติดตามวัตถุขั้นสูง เช่น **ByteTrack** และ **DeepSORT** (มีให้ใช้งานใน Ultralytics YOLO ผ่านคำสั่ง `model.track(tracker="bytetrack.yaml")`) ใช้ตัวกรองคัลมานเพื่อสร้างโมเดลความคลาดเคลื่อนของกรอบล้อมรอบ ($[x, y, a, h, \dot{x}, \dot{y}, \dot{a}, \dot{h}]$ โดยที่ $a$ คืออัตราส่วนภาพ Aspect ratio)
*   **การปรับอัตราส่วนเสียงรบกวนจากการวัด (Measurement Noise Scaling):** ในการวิเคราะห์การกีฬา กล้องมักถูกติดตั้งให้อยู่กับที่ กล้องที่มีความแม่นยำสูงทำให้เราสามารถลดค่าความแปรปรวนร่วม $R$ ลงได้ แต่ถ้าหาก YOLO คำนวณพิกัดกรอบล้อมรอบคลาดเคลื่อนเนื่องจากเงาที่เปลี่ยนแปลงหรือการบิดเบี้ยวของลูกบอล เราสามารถเพิ่มค่า $R$ เพื่อกำหนดให้ตัวติดตามเชื่อใจสมการฟิสิกส์ของการทำนายตำแหน่งที่ราบรื่น (Motion model predictions) มากขึ้นแทน

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX70_Retail_Shelf_Monitoring_TH]]
*   [[EX72_Autonomous_Driving_BEV_TH]]
*   [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   [[learning_journal|บันทึกการเรียนรู้]]
