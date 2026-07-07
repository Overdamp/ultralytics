# 🧠 EX62: การแสดงผลการทำนาย (Inference Visualization)

หลังจากที่โมเดล YOLO ประมวลผลรูปภาพหรือวิดีโอเรียบร้อยแล้ว คุณจำเป็นต้องแสดงผลลัพธ์การทำนายออกมาให้เห็น โดย Ultralytics มีเครื่องมือที่มีมาให้ในตัวอย่างเมธอด `plot()` ซึ่งสามารถนำมาใช้งานร่วมกับ OpenCV หรือ PIL ได้อย่างง่ายดายสำหรับการทำซ้อนทับภาพ (overlays) การวาดรูป และการแสดงผลแบบอินเตอร์แอคทีฟ

---

## 1. การแสดงผลด้วย `results[0].plot()` (Visualization with `results[0].plot()`)

ออบเจกต์ `Results` ที่ได้จากเมธอด predict ของ YOLO จะประกอบด้วยฟังก์ชันอำนวยความสะดวก `plot()` ซึ่งจะช่วยวาดกล่องขอบเขต (bounding box), หน้ากากภาพ (mask), จุดสำคัญ (keypoint), ป้ายกำกับ (label) และค่าความมั่นใจ (probability) ลงบนรูปภาพโดยอัตโนมัติ และส่งคืนค่ากลับมาในรูปแบบอาร์เรย์ NumPy (ฟอร์แมต BGR แบบ `uint8`) ซึ่งสามารถนำไปใช้งานร่วมกับ OpenCV ได้โดยตรง

พารามิเตอร์ที่สำคัญของเมธอด `plot()` มีดังนี้:
*   `conf` (bool): เปิด/ปิดการแสดงคะแนนความมั่นใจ (ค่าเริ่มต้นคือ `True`)
*   `line_width` (int): ความกว้างของเส้นกล่องขอบเขต (หากไม่ระบุจะปรับขนาดอัตโนมัติตามขนาดของภาพ)
*   `font_size` (float): ขนาดตัวอักษรของป้ายกำกับ
*   `labels` (bool): เปิด/ปิดการแสดงชื่อคลาสป้ายกำกับ
*   `boxes` (bool): เปิด/ปิดการแสดงกล่องขอบเขต (มีประโยชน์มากเมื่อต้องการแสดงผลเฉพาะตัวหน้ากากภาพ หรือจุดสำคัญ)
*   `img` (numpy array): ระบุรูปภาพอื่นที่ต้องการให้แสดงทับแทนที่รูปภาพดั้งเดิม

---

## 2. การทำงานในภาษา Python ร่วมกับ OpenCV

ตัวอย่างการทำนายผลลัพธ์ภาพ แสดงผลผ่านเมธอด `plot()` และเปิดหน้าต่างแบบโต้ตอบด้วย OpenCV:

```python
import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO("yolo11n.pt")

# 2. Run inference on a source
results = model("bus.jpg")  # returns a list of Results

# 3. Access the first image result and plot predictions
annotated_frame = results[0].plot(conf=True, line_width=2, labels=True)

# 4. Display the frame using OpenCV
cv2.imshow("YOLO Predictions", annotated_frame)

# 5. Keep window open until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 3. การแสดงผลซ้อนทับแบบกำหนดเองด้วย OpenCV (Customized OpenCV Overlays)

หากคุณต้องการควบคุมรายละเอียดการแสดงผลทั้งหมดด้วยตนเอง เช่น การวาดแดชบอร์ดข้อมูล (HUD - Heads-Up Display), เอฟเฟกต์ความโปร่งใส, หรือรูปแบบตัวหนังสือเฉพาะตัว คุณสามารถดึงข้อมูลรายละเอียดกล่องข้อความออกมาด้วยตนเองและใช้คำสั่งการวาดของ OpenCV ได้ตามใจชอบ:

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
image_path = "bus.jpg"
image = cv2.imread(image_path)

results = model(image_path)
boxes = results[0].boxes

for box in boxes:
    # Get coordinates (pixels)
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Get metadata
    conf = float(box.conf[0].item())
    cls = int(box.cls[0].item())
    label = f"{model.names[cls]}: {conf:.2f}"
    
    # Draw custom bounding box (e.g., green box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Draw a custom filled background label rectangle
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
    
    # Put label text
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

cv2.imshow("Custom Visuals", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 💡 คำแนะนำจากอาจารย์ (Professor Tips)

1.  **ปริภูมิสี RGB เทียบกับ BGR (RGB vs. BGR)**: ไลบรารี OpenCV จะอ่านและบันทึกภาพในรูปแบบ BGR ในขณะที่ PIL จะใช้รูปแบบ RGB โดยทั่วไป เมธอด `results[0].plot()` จะส่งคืนภาพในรูปแบบ BGR (เพื่อให้ใช้งานร่วมกับ `cv2.imshow()` ได้ทันที) แต่หากคุณต้องการส่งต่อไปใช้งานกับ PIL หรือวาดด้วย Matplotlib คุณจะต้องแปลงปริภูมิสีก่อนเสมอ:
    ```python
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    ```
2.  **การปรับประสิทธิภาพในโหมดเรียลไทม์ (Real-Time Optimization)**: การเปิดหน้าต่างแสดงผลด้วย `cv2.imshow()` ในการประมวลผลวิดีโอแบบเรียลไทม์จะสร้างภาระในการประมวลผลเพิ่มขึ้นเป็นอย่างมาก สำหรับการติดตั้งใช้งานบนอุปกรณ์ปลายทาง (Edge Devices เช่น Raspberry Pi หรือ Jetson Nano) ควรกำหนดค่าเพื่อปิดการแสดงผล (`save=False`, `show=False`) เพื่อเร่งการประมวลผลต่อวินาที (FPS) และส่งออกพิกัดดิบไปยังโปรแกรมไคลเอนต์ปลายทางที่เบากว่าแทน
3.  **การทำวัตถุแบบกึ่งโปร่งใส (Alpha Blending)**: ไลบรารี OpenCV ทั่วไปไม่รองรับการวาดรูปทรงโปร่งแสงในตัวโดยตรง หากต้องการทำเช่นนี้ คุณจำเป็นต้องคัดลอกรูปภาพชุดหนึ่งขึ้นมา วาดรูปทรงที่ต้องการลงไป แล้วนำกลับมารวมเข้ากับภาพดั้งเดิมโดยกำหนดค่าน้ำหนักความโปร่งใสผ่านคำสั่ง `cv2.addWeighted()`

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX57_Bounding_Box_Parsing_TH|EX57: การดึงข้อมูลกล่องขอบเขต (Bounding Box Parsing)]]
*   [[EX60_Streaming_Inference_TH|EX60: การตรวจจับแบบสตรีมมิ่ง (Streaming Inference)]]
*   [[EX63_Multi_Object_Tracking_TH|EX63: การติดตามวัตถุหลายชิ้น (Multi-Object Tracking)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal|บันทึกการเรียนรู้]]
