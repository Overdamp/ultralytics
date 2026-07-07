# 🧠 EX72: มุมมองแบบตาเหยี่ยวสำหรับการขับขี่อัตโนมัติ (Bird's Eye View Projection)

ในการขับขี่อัตโนมัติ (Autonomous Driving) กล้องจะจับภาพถนนในลักษณะการฉายภาพแบบทัศนมิติ 3 มิติ (3D Perspective Projection) อย่างไรก็ตาม อัลกอริทึมการวางแผนเส้นทาง การหลบหลีกสิ่งกีดขวาง และการควบคุมรถมักทำงานบนระนาบพื้นดิน 2 มิติ หรือที่เรียกว่า **มุมมองแบบตาเหยี่ยว (Bird's Eye View: BEV)**

เพื่อเชื่อมช่องว่างนี้ เราจึงตรวจจับรถยนต์และสิ่งกีดขวางโดยใช้การตรวจจับวัตถุ 2 มิติ (YOLO) หาจุดสัมผัสที่วัตถุแตะพื้นถนน (จุดกึ่งกลางด้านล่างของกรอบล้อมรอบ) จากนั้นใช้ **การจับคู่ทัศนมิตีย้อนกลับ (Inverse Perspective Mapping: IPM)** เพื่อแปลงพิกัดพิกเซลเหล่านั้นให้อยู่ในตารางพิกัดพื้นดิน 2 มิติในหน่วยเมตร

---

## 1. สูตรทางคณิตศาสตร์และระบบพิกัด

### แบบจำลองการฉายภาพแบบทัศนมิติ (Perspective Projection Model)
ความสัมพันธ์ระหว่างจุดพิกัดโลก 3 มิติ $\mathbf{X}_w = [X_w, Y_w, Z_w]^T$ และจุดพิกัดพิกเซล 2 มิติ $\mathbf{x}_p = [u, v]^T$ ถูกกำหนดโดยเมทริกซ์คุณลักษณะเฉพาะภายในกล้อง (Intrinsic matrix) $\mathbf{K}$ และเมทริกซ์ภายนอกกล้อง (Extrinsic matrix) $[\mathbf{R} | \mathbf{t}]$:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

โดยที่:
*   $\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ คือเมทริกซ์คุณลักษณะเฉพาะภายในของกล้อง (ความยาวโฟกัสและจุดศูนย์กลางทางแสง)
*   $\mathbf{R}$ (เมทริกซ์การหมุน) และ $\mathbf{t}$ (เวกเตอร์การเลื่อนตำแหน่ง) แทนเมทริกซ์ภายนอก ซึ่งระบุตำแหน่งและการวางทิศทางของกล้องสัมพันธ์กับระบบพิกัดถนนหรือพิกัดของตัวรถ
*   $s$ คือค่าตัวคูณสเกลใดๆ ที่แสดงถึงความลึก (Depth)

### การจับคู่ทัศนมิตีย้อนกลับ (Inverse Perspective Mapping: IPM)
เนื่องจากการแปลงพิกเซล 2 มิติกลับไปเป็นพิกัด 3 มิติเป็นปัญหาที่ไม่มีคำตอบที่แน่ชัดทางคณิตศาสตร์ (เนื่องจากข้อมูลความลึกสูญหายไปตามแนวเส้นรังสีของการฉายภาพ) เราจึงต้องเพิ่มเงื่อนไขทางเรขาคณิตเข้ามาช่วย ในสถานการณ์บนท้องถนนเราจะสมมติให้พื้นผิวถนนเป็นพื้นราบ:

$$Z_w = 0$$

กำหนดให้ $\mathbf{M} = \mathbf{K} \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix}$ เป็นเมทริกซ์โฮโมกราฟี (Homography matrix) ขนาด $3\times3$ ที่จับคู่พื้นถนนเข้ากับรูปภาพ โดยที่ $\mathbf{r}_1$ และ $\mathbf{r}_2$ คือสองคอลัมน์แรกของเมทริกซ์ $\mathbf{R}$ เราสามารถเขียนได้ว่า:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{H}_{ground} \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \quad \text{โดยที่} \quad \mathbf{H}_{ground} = \mathbf{K} \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix}$$

เมื่อคำนวณเมทริกซ์โฮโมกราฟีย้อนกลับ (Inverse homography matrix) $\mathbf{H}_{ground}^{-1}$ เราจะสามารถแปลงพิกัดพิกเซลกลับไปยังระนาบถนน 2 มิติได้:

$$\begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \sim \mathbf{H}_{ground}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

---

## 💻 การเขียนโค้ดด้วย Python

สคริปต์ต่อไปนี้ตั้งค่าเมทริกซ์กล้องจำลอง รับผลการทำนายกรอบล้อมรอบรถยนต์จาก YOLO ดึงค่าพิกเซลจุดกึ่งกลางด้านล่างของกรอบล้อมรอบ และฉายภาพจุดนั้นลงบนระนาบพื้นดิน BEV ในหน่วยเมตร

```python
import numpy as np

# 1. Camera Intrinsics (K)
focal_length_px = 800  # fx, fy
img_w, img_h = 1920, 1080
cx, cy = img_w / 2, img_h / 2
K = np.array([
    [focal_length_px, 0, cx],
    [0, focal_length_px, cy],
    [0, 0, 1]
])

# 2. Camera Extrinsics (R, t) - Camera mounted 1.5 meters high, tilted down by 15 degrees
pitch_deg = 15.0
pitch_rad = np.radians(pitch_deg)
c, s = np.cos(pitch_rad), np.sin(pitch_rad)

# Rotation matrix around X-axis (tilting down)
R = np.array([
    [1,  0, 0],
    [0,  c, s],
    [0, -s, c]
])
# Translation vector (1.5 meters above ground)
t = np.array([[0.0], [1.5], [0.0]])

# 3. Form Ground Homography Matrix H_ground (for Z_w = 0)
r1 = R[:, 0:1]
r2 = R[:, 1:2]
H_ground = K.dot(np.hstack((r1, r2, t)))
H_inv = np.linalg.inv(H_ground)

# 4. Simulate YOLO Vehicle Bounding Box Output: [x_min, y_min, x_max, y_max]
# Let's say a vehicle is detected directly in front of the camera
yolo_bbox = [860, 600, 1060, 800]

# Bottom-center of the bounding box represents the contact point on the road
u_bottom = (yolo_bbox[0] + yolo_bbox[2]) / 2.0  # (860+1060)/2 = 960
v_bottom = yolo_bbox[3]                          # 800 (bottom of the box)
pixel_coord = np.array([[u_bottom], [v_bottom], [1.0]])

# 5. Project back to 2D Ground Plane (IPM)
ground_homogeneous = H_inv.dot(pixel_coord)
# Normalize to remove scaling factor s
ground_meters = ground_homogeneous / ground_homogeneous[2, 0]

X_w = ground_meters[0, 0]
Y_w = ground_meters[1, 0]

print(f"YOLO Bounding Box: {yolo_bbox}")
print(f"Road Contact Pixel (u, v): ({u_bottom:.1f}, {v_bottom:.1f})")
print("-" * 50)
print(f"Projected 2D Ground Coordinates (BEV relative to camera base):")
print(f"Lateral Distance (X_w): {X_w:.3f} meters (positive is right, negative is left)")
print(f"Longitudinal Distance (Y_w): {Y_w:.3f} meters (distance ahead on the road)")
```

---

## 💡 เคล็ดลับจากอาจารย์ (Professor Tips)

*   **ปัญหาความลาดเอียงของถนนและอาการโคลงตัว (Road slope and Pitch issues):** อัลกอริทึม IPM มีความอ่อนไหวต่อสมมติฐานเรื่องพื้นดินที่ราบเรียบอย่างมาก หากถนนมีความลาดชัน หรือหากรถยนต์เร่งความเร็วหรือเบรกกะทันหัน (ทำให้เกิดการหมุนในแกน Pitch) ระยะทางที่ฉายได้จะมีความผิดพลาดมหาศาล (เช่น ฉายตำแหน่งรถยนต์ที่ระยะห่างจริง 30 เมตร ออกไปเป็น 50 เมตร) ระบบรถยนต์ไร้คนขับจึงผสานรวม **เซ็นเซอร์วัดความเฉื่อย (IMU)** และเซ็นเซอร์ระบบกันสะเทือนเพื่อปรับปรุงเมทริกซ์การหมุนภายนอกกล้อง $\mathbf{R}$ แบบเรียลไทม์
*   **การเลือกจุดอ้างอิงบนกรอบล้อมรอบ (Bounding Box Anchor Selection):** ในการฉายพิกัดของรถยนต์ ให้เลือกจุด *กึ่งกลางด้านล่าง* ของกรอบล้อมรอบเสมอ หลีกเลี่ยงการใช้จุดศูนย์กลางของกรอบล้อมรอบ เนื่องจากจุดศูนย์กลางลอยอยู่ในอากาศเหนือพื้นดิน ซึ่งเป็นการขัดต่อสมมติฐาน $Z_w = 0$ และจะส่งผลให้วัตถุถูกฉายตำแหน่งออกไปไกลเกินจริงอย่างมาก

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX71_Sports_Analytics_Tracking_TH]]
*   [[EX73_Smart_Agriculture_Grading_TH]]
*   [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   [[learning_journal|บันทึกการเรียนรู้]]
