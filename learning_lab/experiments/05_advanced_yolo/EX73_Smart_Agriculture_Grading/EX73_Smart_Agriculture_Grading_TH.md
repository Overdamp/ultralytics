# 🧠 EX73: การคัดเกรดผลผลิตทางการเกษตรอัจฉริยะ (Instance Segmentation & Color Analysis)

ในระบบการเกษตรอัจฉริยะ (Smart Agriculture) ระบบคัดเกรดอัตโนมัติจะคัดแยกผลผลิตโดยอิงจากคุณลักษณะทางกายภาพ 2 ประการ ได้แก่ **ขนาด** (ปริมาตร/เส้นผ่านศูนย์กลาง) และ **ระดับความสุก** (สีสันของเปลือก)

การใช้กรอบล้อมรอบ (Bounding boxes) มาตรฐานนั้นไม่เพียงพอสำหรับการคัดเกรด เนื่องจากกรอบล้อมรอบจะรวมพิกเซลพื้นหลังเข้าไปด้วยและไม่สามารถตรวจวัดวัตถุที่มีรูปทรงไม่สม่ำเสมอได้ ดังนั้น เราจึงเลือกใช้ **YOLO Instance Segmentation (YOLO-seg)** เพื่อหาหน้ากากพิกเซล (Pixel-level masks) ที่แม่นยำ จากนั้นจึงคำนวณขนาดทางกายภาพจริงจากหน้ากากไบนารีเหล่านี้ และแปลงปริภูมิสีเป็น **HSL (Hue, Saturation, Lightness)** เพื่อทำการประเมินเกรดความสุกได้อย่างแม่นยำและทนทานต่อการเปลี่ยนแปลงของแสง

---

## 1. หลักการวิเคราะห์ขนาดและสี

### ขนาดที่แม่นยำจากหน้ากากไบนารี (Binary Masks)
หน้ากากแบ่งส่วน (Segmentation mask) คือเมทริกซ์ไบนารี $M(u,v) \in \{0, 1\}$ ที่แทนพิกเซลของวัตถุ เราสามารถคำนวณได้ดังนี้:
*   **พื้นที่ในหน่วยพิกเซล ($A_{pixel}$):** ผลรวมของพิกเซลที่เป็นบวกในหน้ากาก:
    $$A_{pixel} = \sum_{u} \sum_{v} M(u,v)$$
*   **การแปลงค่าขนาดทางกายภาพจริง (Metric Size Calibration):** ใช้มาตราส่วนอ้างอิงที่ทราบค่า (ค่าปรับเทียบ $C = \text{mm/pixel}$):
    $$\text{Physical Area (mm}^2) = A_{pixel} \times C^2$$
    $$\text{Physical Diameter (mm)} = \text{Pixel Diameter} \times C$$

### ปริภูมิสี HSL สำหรับการประเมินความสุก (Ripeness)
ค่าสี RGB มีความอ่อนไหวสูงต่อการเปลี่ยนแปลงของแสงเงาและแสงสะท้อน ส่วนปริภูมิสี **HSL (Hue, Saturation, Lightness)** จะแยกส่วนของประเภทสี (Chrominance) ออกจากความสว่าง (Luminance):
*   **โทนสี (Hue: $H \in [0^\circ, 360^\circ]$):** แทนค่าสีแท้จริง เช่น:
    *   $60^\circ - 150^\circ$ ตรงกับ **สีเขียว** (ดิบ / Unripe)
    *   $15^\circ - 60^\circ$ ตรงกับ **สีเหลือง/ส้ม** (สุก / Ripe)
    *   $0^\circ - 15^\circ$ หรือ $330^\circ - 360^\circ$ ตรงกับ **สีแดง** (สุกงอม / Overripe)
*   **ความอิ่มตัวของสี (Saturation: $S$):** ความเข้มหรือความบริสุทธิ์ของสี
*   **ความสว่าง (Lightness: $L$):** ระดับความสว่างของสี

การวิเคราะห์ฮิสโตแกรมของช่องสี Hue *เฉพาะ* บริเวณพิกเซลที่ $M(u,v) = 1$ ช่วยให้เราจำแนกเกรดความสุกของผลไม้ได้อย่างแม่นยำ โดยไม่ถูกรบกวนจากการเปลี่ยนแปลงของระดับความเข้มแสงภายนอก

---

## 💻 การเขียนโค้ดด้วย Python

สคริปต์ต่อไปนี้จำลองการรับข้อมูลหน้ากากไบนารี YOLO-seg และภาพสี RGB ของมะม่วงที่ถูกครอบตัด จากนั้นคำนวณขนาดทางกายภาพจริง และหาเกรดความสุกด้วยการวิเคราะห์ช่องสี Hue

```python
import numpy as np
import matplotlib.colors as colors

# 1. Generate synthetic fruit data (a yellow-green mango mask & image)
height, width = 200, 200
Y, X = np.ogrid[:height, :width]
center_y, center_x = 100, 100
radius = 60

# Binary mask (circle)
mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2

# RGB Image: Yellow-green colors inside the mask, black outside
rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
# Ripe yellow-orange is roughly [230, 180, 50], unripe green is [80, 180, 50]
# We'll fill the circle with a gradient representing ripening progress
for y in range(height):
    for x in range(width):
        if mask[y, x]:
            # Gradient: left side is yellow, right side is green
            ratio = x / width
            r = int(230 * (1 - ratio) + 80 * ratio)
            g = int(180)
            b = int(50)
            rgb_image[y, x] = [r, g, b]

# 2. Compute Physical Dimensions
calibration_factor = 0.25  # 1 pixel = 0.25 mm
pixel_area = np.sum(mask)
physical_area_mm2 = pixel_area * (calibration_factor ** 2)

# Diameter calculation (bounding box width from mask)
mask_coords = np.argwhere(mask)
y_min, x_min = mask_coords.min(axis=0)
y_max, x_max = mask_coords.max(axis=0)
pixel_diameter = max(y_max - y_min, x_max - x_min)
physical_diameter_mm = pixel_diameter * calibration_factor

# 3. HSL Color Analysis
# Convert RGB to normalized float [0, 1], then to HSL
rgb_normalized = rgb_image / 255.0
hsv_image = colors.rgb_to_hsv(rgb_normalized)  # In matplotlib/numpy, HSV is commonly used (Hue is equivalent to HSL Hue)

# Extract Hue values for mask pixels only
hue_values = hsv_image[:, :, 0][mask] * 360.0  # Convert scale 0-1 to degrees 0-360

# Classify ripeness based on mean Hue
mean_hue = np.mean(hue_values)

if 60.0 <= mean_hue <= 150.0:
    ripeness_grade = "Unripe (Green)"
elif 20.0 <= mean_hue < 60.0:
    ripeness_grade = "Ripe (Yellow/Orange)"
else:
    ripeness_grade = "Overripe (Red)"

print(f"--- Fruit Grading Report ---")
print(f"Total Pixel Area: {pixel_area} pixels")
print(f"Physical Area: {physical_area_mm2:.2f} mm²")
print(f"Physical Diameter: {physical_diameter_mm:.2f} mm")
print(f"Average Hue Angle: {mean_hue:.1f}°")
print(f"Ripeness Classification: {ripeness_grade}")
```

---

## 💡 เคล็ดลับจากอาจารย์ (Professor Tips)

*   **ตัวอ้างอิงการปรับเทียบ (Calibration Markers):** ในสายพานลำเลียงของโรงงานจริง ระยะห่างของกล้องจะถูกยึดไว้คงที่ การหาพารามิเตอร์การคัดเทียบกล้อง (Camera Calibration) จะทำเพียงครั้งเดียวโดยใช้ **ตัวระบุ ArUco (ArUco markers)** หรือแผ่นตารางหมากรุก เพื่อคำนวณความสัมพันธ์ทางโฮโมกราฟีและระยะทางกายภาพต่อพิกเซลที่แน่นอน
*   **ทำไมช่องสี Hue ถึงไม่ขึ้นกับความสว่าง:** เงาที่พาดผ่านผลผลิตเนื่องจากความโค้งมนจะทำให้ค่าสี RGB ลดลงเป็นสัดส่วนเท่าๆ กัน แต่เนื่องจากค่า Hue คำนวณในรูปมุมองศา ($H = \arctan(\dots)$ ของผลต่างสี) การคูณสเกล R, G, B ด้วยสัมประสิทธิ์ความเข้มแสงใดๆ จึงไม่เปลี่ยนมุมองศาของ Hue ส่งผลให้การตรวจจับสีมีความเสถียรภายใต้สภาพแสงที่ไม่สม่ำเสมอ

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX72_Autonomous_Driving_BEV_TH]]
*   [[EX74_Thermal_Fire_Detection_TH]]
*   [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   [[learning_journal|บันทึกการเรียนรู้]]
