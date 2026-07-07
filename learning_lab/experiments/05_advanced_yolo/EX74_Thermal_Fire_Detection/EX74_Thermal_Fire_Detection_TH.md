# 🧠 EX74: การตรวจจับอัคคีภัยด้วยกล้องความร้อน (Dual-Camera Sensor Fusion)

การใช้กล้อง RGB เพียงอย่างเดียวในการตรวจจับไฟและควันมักมีแนวโน้มที่จะเกิดการแจ้งเตือนเท็จ (False Positives) ได้ง่ายมาก เนื่องจากลักษณะภายนอกที่เป็นสีส้มของเสื้อสะท้อนแสงเพื่อความปลอดภัย ไฟฉุกเฉินสีแดง แสงสะท้อนของดวงอาทิตย์ หรือป้ายโฆษณาสีเหลือง สามารถหลอกโมเดล YOLO ให้สั่งการระบบแจ้งเตือนภัยอย่างผิดพลาดได้ง่าย

เพื่อสร้างระบบแจ้งเตือนอัจฉริยะที่มีความน่าเชื่อถือสูงในอุตสาหกรรม เราจึงนำเทคนิค **การผสานรวมเซ็นเซอร์กล้องคู่จัดตำแหน่งร่วม (Co-registered dual-camera sensor fusion)** มาใช้ โดยผสมผสานวิดีโอ RGB ความละเอียดสูงเข้ากับกล้องความร้อนอินฟราเรดคลื่นยาว (LWIR) ขั้นตอนการทำงานเริ่มจากการรันโมเดล YOLO บนสตรีมภาพ RGB จากนั้นฉายกรอบล้อมรอบ (Bounding box) ที่ตรวจพบไปยังตารางพิกเซลของกล้องความร้อนโดยใช้ **เมทริกซ์โฮโมกราฟี (Homography Matrix)** แล้วทำการวิเคราะห์ค่าความร้อนภายในพื้นที่ดังกล่าวเพื่อยืนยันว่ามีแหล่งความร้อนสูงอยู่จริงหรือไม่

---

## 1. หลักการจัดตำแหน่งร่วมและการผสานรวมด้วยโฮโมกราฟี (Homography Fusion)

### ระยะเยื้องของกล้อง (พาราแลกซ์) และโฮโมกราฟี (Camera Offset and Homography)
เนื่องจากเลนส์ของกล้อง RGB และกล้องความร้อนติดตั้งอยู่คนละตำแหน่งพิกัดทางกายภาพ มุมมองของกล้องทั้งสองจึงไม่ตรงกันพอดี ระยะเยื้องของตำแหน่งนี้เรียกว่า **พาราแลกซ์ (Parallax)**

หากฉากหลังค่อนข้างแบนราบหรืออยู่ห่างจากกล้องมากพอ เราสามารถสร้างแบบจำลองการแปลงพิกัดระหว่างระนาบภาพทั้งสองได้โดยใช้ **เมทริกซ์โฮโมกราฟี (Homography Matrix)** ขนาด $3\times3$ สัญลักษณ์ $\mathbf{H}$:

$$\mathbf{x}_{thermal} \sim \mathbf{H} \mathbf{x}_{rgb}$$

เขียนให้อยู่ในรูปพิกัดเอกพันธุ์ (Homogeneous coordinates) ได้ดังนี้:

$$\begin{bmatrix} x_t \\ y_t \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_r \\ y_r \\ 1 \end{bmatrix}$$

เมื่อนำการฉายภาพมาคำนวณจะได้:

$$x_t = \frac{h_{11}x_r + h_{12}y_r + h_{13}}{h_{31}x_r + h_{32}y_r + h_{33}}, \quad y_t = \frac{h_{21}x_r + h_{22}y_r + h_{23}}{h_{31}x_r + h_{32}y_r + h_{33}}$$

### ขั้นตอนการทำงานสำหรับการตรวจสอบการแจ้งเตือน (Alarm Verification Pipeline)
1. **การตรวจจับ (Detection):** รัน YOLO บนเฟรม RGB เพื่อค้นหาวัตถุเป้าหมายที่เข้าข่าย เช่น `fire` (ไฟ) หรือ `smoke` (ควัน) โดยส่งคืนพิกัดกรอบล้อมรอบ $[x_1, y_1, x_2, y_2]$
2. **การฉายพิกัด (Coordinate Projection):** ฉายจุดมุมของกรอบล้อมรอบ RGB เข้าสู่พิกัดของกล้องความร้อนโดยใช้เมทริกซ์ $\mathbf{H}$
3. **การดึงค่าความร้อน (Thermal Query):** ดึงค่าความเข้มของพิกเซลความร้อน (Thermal pixel values) จากพื้นที่ที่ถูกฉายในเฟรม LWIR
4. **กฎการตัดสินใจ (Decision Rules):** ตรวจสอบว่าค่าอุณหภูมิสูงสุดภายในพื้นที่ดังกล่าวสูงเกินเกณฑ์อุณหภูมิความปลอดภัยที่ตั้งไว้หรือไม่ ($T_{threshold} \ge 80^\circ\text{C}$) หากไม่ถึงเกณฑ์ ให้ระบุว่าเป็นการแจ้งเตือนเท็จและยกเลิกการส่งสัญญาณเตือน

---

## 💻 การเขียนโค้ดด้วย Python

สคริปต์ต่อไปนี้จำลองกระบวนการทำงานของการผสานรวมข้อมูลเซ็นเซอร์: ฉายกรอบล้อมรอบที่ตรวจจับได้จาก RGB ไปยังตารางภาพความร้อนที่ถูกจัดตำแหน่งร่วมกัน และตรวจสอบสถิติค่าอุณหภูมิ

```python
import numpy as np

# 1. Define the RGB-to-Thermal Homography Matrix (H)
# Calculated during camera calibration
H = np.array([
    [0.85, 0.02, 10.0],
    [-0.01, 0.84, 15.0],
    [0.0001, -0.0002, 1.0]
])

def project_point(x, y, homography):
    """Projects a single 2D point from RGB to Thermal coordinates."""
    point = np.array([[x], [y], [1.0]])
    projected = homography.dot(point)
    # Convert from homogeneous coordinates
    x_proj = projected[0, 0] / projected[2, 0]
    y_proj = projected[1, 0] / projected[2, 0]
    return int(round(x_proj)), int(round(y_proj))

# 2. Simulate YOLO RGB Detection of a "fire" candidate
# Bounding box coordinates: [x_min, y_min, x_max, y_max]
rgb_bbox = [200, 150, 300, 250]

# Project the top-left and bottom-right corners to the thermal frame
tx_min, ty_min = project_point(rgb_bbox[0], rgb_bbox[1], H)
tx_max, ty_max = project_point(rgb_bbox[2], rgb_bbox[3], H)

print(f"YOLO RGB Detection Bounding Box: {rgb_bbox}")
print(f"Projected Thermal Bounding Box : [{tx_min}, {ty_min}, {tx_max}, {ty_max}]")
print("-" * 65)

# 3. Simulate two scenarios for the corresponding thermal image crop (e.g. 400x400 grid)
# Scenario A: False alarm (a hot lamp or red balloon - visual matches, but temperature is 35°C)
thermal_grid_A = np.random.uniform(20.0, 35.0, size=(400, 400)) 

# Scenario B: True fire (temperature in the center of the crop reaches 250°C)
thermal_grid_B = np.random.uniform(20.0, 35.0, size=(400, 400))
# Inject heat values in the projected area
thermal_grid_B[ty_min:ty_max, tx_min:tx_max] = np.random.uniform(150.0, 300.0, size=(ty_max-ty_min, tx_max-tx_min))

# 4. Verification Logic
temp_threshold_celsius = 80.0

for name, grid in [("Scenario A (Red Balloon)", thermal_grid_A), ("Scenario B (Actual Fire)", thermal_grid_B)]:
    # Crop the projected region from the thermal grid
    # Ensure crop coordinates do not exceed grid boundaries
    h_max, w_max = grid.shape
    crop = grid[max(0, ty_min):min(h_max, ty_max), max(0, tx_min):min(w_max, tx_max)]
    
    max_temp = np.max(crop)
    mean_temp = np.mean(crop)
    
    print(f"\nEvaluating {name}:")
    print(f" - Max Crop Temp: {max_temp:.1f}°C")
    print(f" - Mean Crop Temp: {mean_temp:.1f}°C")
    
    if max_temp >= temp_threshold_celsius:
        print(f"🔥 ALARM CONFIRMED: High temperature signature verified.")
    else:
        print(f"🟢 FALSE POSITIVE DISMISSED: Temperature within normal limits.")
```

---

## 💡 เคล็ดลับจากอาจารย์ (Professor Tips)

*   **ข้อจำกัดของพาราแลกซ์ (Parallax Limits):** การหาความสัมพันธ์ด้วยโฮโมกราฟีจะให้ผลลัพธ์ที่แม่นยำสูงก็ต่อเมื่อวัตถุเป้าหมายทั้งหมดอยู่บนระนาบเดียวกัน (เช่น ระนาบพื้นดิน) หรืออยู่ห่างจากชุดกล้องมากๆ (ซึ่งจะทำให้ระยะห่างทางกายภาพระหว่างจุดกึ่งกลางของเลนส์ทั้งสองกลายเป็นศูนย์โดยปริยาย) แต่สำหรับวัตถุเป้าหมายที่อยู่ใกล้กล้องและมีความสูง-ลึกที่แตกต่างกันอย่างมาก การใช้โฮโมกราฟีจะมีความคลาดเคลื่อนสูง ในกรณีนั้น นักพัฒนาจำเป็นต้องเลือกใช้กล้องวัดความลึก (Depth Sensors) หรืออัลกอริทึมการจับคู่สเตอริโอ (Stereo Matching) เพื่อช่วยจัดตำแหน่งพิกเซลของภาพ RGB และ LWIR แบบพิกเซลต่อพิกเซล
*   **การปรับเทียบค่าดิบของกล้องความร้อน (Thermal Raw Data Calibration):** กล้องความร้อนอุตสาหกรรมส่วนใหญ่จะส่งคืนข้อมูลพิกเซลดิบขนาด 14 บิต (Radiometric data) แทนที่จะส่งข้อมูลเป็นองศาเซลเซียสโดยตรง คุณจะต้องทำการแปลงรหัสพิกเซลดิบนี้ให้เป็นองศาเคลวินหรือองศาเซลเซียสก่อนเสมอโดยใช้พารามิเตอร์จำเพาะของกล้อง (เช่น ค่าความแผ่รังสีของวัตถุ Emissivity, อุณหภูมิโดยรอบ และระยะห่างจากวัตถุ) ก่อนนำไปเข้าสู่เงื่อนไขกฎการตัดเกณฑ์ความร้อน

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX73_Smart_Agriculture_Grading_TH]]
*   [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   [[learning_journal|บันทึกการเรียนรู้]]
