# 🧠 EX63: การติดตามวัตถุหลายชิ้น (Multi-Object Tracking - MOT)

ในขณะที่การตรวจจับวัตถุทั่วไป (object detection) จะหาพิกัดและตำแหน่งของวัตถุแบบเฟรมต่อเฟรมโดยไม่สนใจความเกี่ยวเนื่องด้านเวลา การติดตามวัตถุหลายชิ้น (Multi-Object Tracking) จะช่วยแก้ปัญหานี้โดยการกำหนดค่า ID เฉพาะตัวและคงอยู่ตลอดเวลา (persistent ID) ให้กับแต่ละวัตถุที่ตรวจจับได้ พร้อมทั้งติดตามแนววิถีการเคลื่อนที่ (trajectory) ของวัตถุนั้นๆ ข้ามเฟรมวิดีโออย่างต่อเนื่อง

---

## 1. ความเชื่อมโยงเชิงเวลาและการใช้ `model.track()` (Temporal Association and `model.track()`)

สำหรับการทำงานใน Ultralytics YOLO กระบวนการติดตามวัตถุจะถูกเรียกใช้ผ่านเมธอด `model.track()` โดยเบื้องหลังแล้ว YOLO จะรันการตรวจจับวัตถุในแต่ละเฟรมก่อน จากนั้นจะส่งข้อมูลกล่องขอบเขตไปให้ตัวอัลกอริทึมติดตามวัตถุ (tracking algorithm) ทำการเชื่อมโยงข้อมูลเชิงเวลา (temporal association) โดยอาศัยรูปแบบการเคลื่อนที่ (ด้วย Kalman Filters) และข้อมูลลักษณะทางกายภาพ (ด้วย Re-Identification networks)

พารามิเตอร์หลักของเมธอด `model.track()` มีดังนี้:
*   `source`: แหล่งข้อมูลวิดีโออินพุต, พาธ URL ของสตรีมมิ่ง, กล้องเว็บแคม หรือโฟลเดอร์ภาพนิ่ง
*   `tracker`: ระบุตัวอัลกอริทึมติดตามวัตถุที่ต้องการใช้งาน (`'botsort.yaml'` หรือ `'bytetrack.yaml'`)
*   `persist`: (bool) กำหนดเป็น `True` เมื่อประมวลผลเฟรมวิดีโออย่างต่อเนื่องในลูป เพื่อให้ตัวติดตามสามารถจดจำสถานะเฟรมก่อนหน้าได้

---

## 2. ตัวอัลกอริทึมติดตามวัตถุ: BoT-SORT เทียบกับ ByteTrack (Tracking Algorithms)

YOLO มีระบบติดตามวัตถุระดับแนวหน้า (state-of-the-art) 2 ตัวที่คอนฟิกไว้ให้พร้อมใช้งานทันที:

| คุณลักษณะ | BoT-SORT (`botsort.yaml`) | ByteTrack (`bytetrack.yaml`) |
| :--- | :--- | :--- |
| **ตัวเลือกเริ่มต้น** | ใช่ (ค่าเริ่มต้นของ YOLO) | ไม่ใช่ |
| **โมเดลวิถีเคลื่อนที่** | Kalman Filter + การชดเชยการเคลื่อนที่ของกล้อง (CMC) | Kalman Filter |
| **การจำแนกลักษณะ (Re-ID)**| ใช่ (เชื่อมโยงวัตถุด้วยรูปลักษณะภายนอก) | ไม่ใช่ (อิงจากรูปแบบการเคลื่อนที่/พื้นที่ทับซ้อน IoU เท่านั้น) |
| **ความเร็วในการรัน** | ปานกลาง (ใช้พลังการคำนวณมากกว่า) | สูง (ทำงานได้รวดเร็วและเบากว่า) |
| **จุดเด่นหลัก** | ทนทานต่อการขยับของกล้องและการบดบังวัตถุเป็นเวลานาน | เหมาะสำหรับกล้องมุมนิ่งที่มีฝูงชนหนาแน่นเคลื่อนไหวเร็ว |

คุณสามารถปรับแต่งไฮเปอร์พารามิเตอร์ของตัวติดตาม (เช่น เกณฑ์ค่าความมั่นใจในการจับคู่) ได้โดยแก้ไของค์ประกอบในไฟล์คอนฟิกูเรชันโดยตรง หรือชี้พาธไปยังไฟล์ `.yaml` คอนฟิกที่กำหนดเองได้

---

## 3. การดึงข้อมูลรหัสติดตามวัตถุ (Track IDs) ด้วย Python

เมื่อเปิดใช้งานฟังก์ชันการติดตาม อ็อบเจกต์ `Results` จะเก็บคุณลักษณะ `.id` ไว้ภายใต้ฟิลด์ `boxes` เนื่องจากบางเฟรมอาจไม่มีวัตถุที่ติดตามอยู่เลย คุณจึงต้องตรวจสอบก่อนว่ามีรหัสติดตามอยู่หรือไม่ ก่อนจะดึงข้อมูลเหล่านั้นออกมาใช้งาน

ด้านล่างนี้คือตัวอย่างการเขียนโค้ดเต็มรูปแบบ:

```python
import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO("yolo11n.pt")

# 2. Open a video source
cap = cv2.VideoCapture("traffic.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # 3. Run tracking on the frame. Persist must be True for video streams.
    # We use ByteTrack for high performance/frame rate
    results = model.track(source=frame, tracker="bytetrack.yaml", persist=True)
    
    # 4. Check if any tracking IDs were assigned in this frame
    if results[0].boxes.id is not None:
        # Extract coordinates, tracking IDs, and class indices
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()
        
        # Iterate through detections and print track details
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            class_name = model.names[class_id]
            print(f"ID {track_id}: {class_name} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
    # Optional: Render tracker visual annotations
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 💡 คำแนะนำจากอาจารย์ (Professor Tips)

1.  **ทำไมรหัส ID ของฉันรีเซ็ตใหม่ตลอดเวลา?**: ตัวติดตามต้องการความต่อเนื่องเชิงเวลาในการประมวลผล หากคุณรันการติดตามทีละภาพโดยไม่ได้ตั้งค่า `persist=True` ตัวติดตามจะเริ่มต้นประมวลผลใหม่ตั้งแต่ศูนย์ในทุกๆ เฟรม ส่งผลให้ ID ของวัตถุทั้งหมดถูกรีเซ็ตเริ่มจาก $1$ เสมอ
2.  **การจัดการเมื่อวัตถุถูกบดบัง (Handling Occlusions)**: ความท้าทายที่สำคัญในการทำ MOT คือ การที่วัตถุถูกบดบังชั่วคราว (เช่น คนเดินผ่านหลังเสาไฟหรือต้นไม้) ตัวติดตาม ByteTrack มีจุดเด่นด้านนี้โดยจะเก็บรักษาประวัติของสถานะติดตามที่หายไป ("lost") ไว้ในบัฟเฟอร์ช่วงระยะเวลาหนึ่ง และพยายามจับคู่พวกมันเข้ากับกล่องขอบเขตที่มีค่าความมั่นใจต่ำลงในเฟรมถัดๆ ไป ก่อนที่จะลบรหัส ID นั้นทิ้งอย่างถาวร
3.  **การตั้งค่าตัวติดตามเอง (Custom Tracker Configurations)**: คุณสามารถเข้าไปดูค่าคอนฟิกเริ่มต้นของตัวติดตามได้ในไดเรกทอรี `ultralytics/cfg/trackers/` คัดลอกไฟล์เหล่านั้น (เช่น `bytetrack.yaml`) ออกมาไว้ในพื้นที่ทำงานของคุณ จากนั้นแก้ไขพารามิเตอร์ เช่น `track_buffer` (จำนวนเฟรมที่จะรอการจับคู่วัตถุที่หายไป) หรือ `match_thresh` และสามารถเรียกใช้งานได้ดังนี้:
    ```python
    model.track(source, tracker="custom_bytetrack.yaml")
    ```

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX60_Streaming_Inference_TH|EX60: การตรวจจับแบบสตรีมมิ่ง (Streaming Inference)]]
*   [[EX62_Inference_Visualization_TH|EX62: การแสดงผลการทำนาย (Inference Visualization)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal|บันทึกการเรียนรู้]]
