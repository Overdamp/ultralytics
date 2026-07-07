# 🧠 EX56: แหล่งข้อมูลสำหรับการพยากรณ์ด้วย YOLO (YOLO Prediction Sources)

ในการประยุกต์ใช้คอมพิวเตอร์วิทัศน์ (Computer Vision) ในโลกแห่งความเป็นจริง การทำนายผลของโมเดล (Inference) มักจะไม่ได้จำกัดอยู่เพียงแค่การประมวลผลไฟล์ภาพนิ่งเท่านั้น ระบบที่ใช้งานจริงจะต้องสามารถรับข้อมูลจากแหล่งที่มาที่หลากหลาย เช่น ไฟล์ภาพนิ่งในเครื่อง บัฟเฟอร์ไฟล์วิดีโอ อาร์เรย์ในหน่วยความจำ กล้องถ่ายภาพแบบเรียลไทม์ หรือแม้กระทั่งการสตรีมผ่านเครือข่าย

Ultralytics YOLO ช่วยลดความยุ่งยากเหล่านี้โดยการจัดหาตัวประมวลผลอินพุตแบบรวมศูนย์ผ่านฟังก์ชัน `model.predict()` โดยเบื้องหลังแล้ว YOLO จะตรวจสอบประเภทอินพุตและสร้างตัวโหลดข้อมูล (data loader) ที่เหมาะสมขึ้นมาโดยอัตโนมัติ เพื่อทำหน้าที่ถอดรหัส (decoding) ปรับขนาดภาพ (resizing) ปรับข้อมูลให้เป็นมาตรฐาน (normalization) และแปลงให้เป็นเทนเซอร์ (tensor conversion)

---

## 1. เบื้องหลังการทำงาน: ท่อส่งข้อมูลการทำนายผล (Inference Data Pipeline)

เมื่อคุณส่งแหล่งข้อมูลอินพุต (source) ให้กับ `model.predict()` ทาง YOLO จะดำเนินกระบวนการเตรียมข้อมูลล่วงหน้า (pre-processing) หลายขั้นตอนดังนี้:
1. **การตรวจสอบแหล่งข้อมูล (Source Identification):** ตรวจสอบว่าแหล่งข้อมูลเป็นเส้นทางไฟล์ (file path) ไดเรกทอรี ลิงก์ URL อาร์เรย์ NumPy หรืออ็อบเจกต์ภาพ PIL
2. **การถอดรหัสและการโหลดข้อมูล (Decoding & Loading):** ใช้ไลบรารีอย่าง OpenCV (`cv2`) สำหรับการถอดรหัสวิดีโอ/เว็บแคม ใช้ PIL สำหรับการอ่านไฟล์รูปภาพ หรือเข้าถึงหน่วยความจำโดยตรงสำหรับตัวแปรอาร์เรย์
3. **การปรับขนาดและการทำเล็ตเตอร์บ็อกซ์ (Resizing & Letterboxing):** ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ (เช่น $640 \times 640$) โดยยังคงสัดส่วนภาพเดิมไว้โดยการเพิ่มขอบสีดำหรือสีเทา (Padding หรือ Letterboxing)
4. **การเปลี่ยนลำดับมิติและการปรับค่าข้อมูล (Permutation & Normalization):** แปลงภาพสีขนาดทั่วไป $H \times W \times C$ (สูง, กว้าง, ช่องสี) ให้เป็นเทนเซอร์ของ PyTorch ในรูปแบบมิติ $C \times H \times W$ พร้อมทั้งปรับระดับค่าสีจากเดิม $[0, 255]$ ให้อยู่ในช่วง $[0.0, 1.0]$ และจัดกลุ่มข้อมูลเป็นแบทช์ ($B \times C \times H \times W$)

---

## 2. ตัวอย่างการเขียนโค้ดด้วย Python (Python Code Demonstration)

ตัวอย่างด้านล่างนี้แสดงวิธีการรันการทำนายผลผ่าน Python API ด้วยแหล่งข้อมูลอินพุตประเภทต่างๆ:

```python
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. โหลดโมเดล YOLO ที่ฝึกสอนไว้ล่วงหน้า
model = YOLO("yolo11n.pt")  # โหลดโมเดล YOLO11 Nano ขนาดเล็กและเบา

# 2. ไฟล์ในเครื่องและไดเรกทอรี (Local Files & Directories)
# ระบุพาธรูปภาพเดี่ยว (string หรือ pathlib.Path)
results_img = model.predict(source="demo.jpg", save=True)

# ระบุไดเรกทอรีที่มีรูปภาพหลายไฟล์
results_dir = model.predict(source="path/to/images_dir/", save=True)

# ใช้รูปแบบ Glob filter เพื่อเลือกเฉพาะนามสกุลไฟล์ที่ต้องการ
results_glob = model.predict(source="path/to/images_dir/*.png", save=True)

# 3. แหล่งข้อมูลวิดีโอ (Video Sources)
# ไฟล์วิดีโอภายในเครื่อง (.mp4, .avi, .mov เป็นต้น)
results_vid = model.predict(source="traffic.mp4", save=True)

# 4. อ็อบเจกต์ในหน่วยความจำ (In-Memory Processing)
# ภาพจากไลบรารี PIL
pil_img = Image.open("demo.jpg")
results_pil = model.predict(source=pil_img)

# อาร์เรย์ NumPy (ฟอร์แมต OpenCV: BGR)
numpy_img = cv2.imread("demo.jpg")
results_cv2 = model.predict(source=numpy_img)

# อาร์เรย์ NumPy ที่จำลองขึ้นมา (เช่น บัฟเฟอร์เฟรมดิบ)
# ขนาดรูปร่าง: (H, W, C) -> (480, 640, 3)
random_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
results_random = model.predict(source=random_frame)

# 5. การสตรีมสดและอุปกรณ์ฮาร์ดแวร์ (Live Streams & Hardware)
# เว็บแคมในเครื่อง (ปกติเลข 0 คือกล้องหลักของระบบ)
# กดปุ่ม 'q' บนแป้นพิมพ์เพื่อปิดหน้าต่างสตรีมหากเปิดใช้งาน show=True
results_webcam = model.predict(source=0, show=True)

# สตรีมสดจาก YouTube
# จำเป็นต้องติดตั้ง: pip install yt-dlp
youtube_url = "https://www.youtube.com/watch?v=LNwODJXard4"
results_youtube = model.predict(source=youtube_url, show=True)
```

---

## 3. การใช้งานผ่าน CLI (CLI Command Equivalence)

Command Line Interface (CLI) เหมาะสำหรับการทดสอบผลลัพธ์และทำนายพิกัดอย่างรวดเร็ว โดยไม่จำเป็นต้องเขียนสคริปต์ Python:

```bash
# ทำนายผลบนภาพเดี่ยวและบันทึกภาพผลลัพธ์
yolo detect predict model=yolo11n.pt source="demo.jpg" save=True

# ทำนายผลบนรูปภาพทั้งหมดในไดเรกทอรี
yolo detect predict model=yolo11n.pt source="path/to/images_dir/" save=True

# รันการทำนายผลบนไฟล์วิดีโอ
yolo detect predict model=yolo11n.pt source="traffic.mp4" save=True

# แสดงสตรีมการทำนายผลจากกล้องเว็บแคมโดยตรง
yolo detect predict model=yolo11n.pt source=0 show=True

# แสดงการสตรีมและทำนายผลจากไลฟ์สด YouTube (ต้องติดตั้ง yt-dlp)
yolo detect predict model=yolo11n.pt source="https://www.youtube.com/watch?v=LNwODJXard4" show=True
```

---

## 💡 คำแนะนำจากอาจารย์ (Professor Tips)

### 1. เปรียบเทียบอาร์เรย์ในหน่วยความจำกับเส้นทางไฟล์ (In-Memory Arrays vs. Disk Paths)
เมื่อคุณส่งอินพุตในหน่วยความจำ เช่น อ็อบเจกต์ PIL หรืออาร์เรย์ NumPy โมเดลจะข้ามขั้นตอนการอ่านข้อมูลจากฮาร์ดดิสก์ ซึ่งเหมาะอย่างยิ่งสำหรับการทำบริการหลังบ้านในระบบโปรดักชัน (เช่น FastAPI หรือ Flask) ที่รับรูปภาพผ่านทางเครือข่าย อย่างไรก็ตามควรคำนึงถึงสิ่งต่อไปนี้:
* ส่งค่า **อาร์เรย์ดิบ/ภาพ PIL** หากคุณมีการปรับแต่งแก้ไขรูปเฟรมนั้นๆ อยู่แล้วในสคริปต์ Python (เช่น การใส่ฟิลเตอร์หรือตัดส่วนภาพ)
* ส่งค่า **เส้นทางไฟล์ (File Paths)** หากต้องการให้ระบบจัดการโหลดข้อมูลของ YOLO ซึ่งถูกออกแบบให้ทำงานแบบมัลติเธรดในการอ่านข้อมูลจากดิสก์บนเธรดเบื้องหลังพร้อมๆ กัน

### 2. การข้ามเฟรมวิดีโอ (`vid_stride`)
เมื่อคุณต้องประมวลผลวิดีโอที่มีความยาวมากหรือสตรีมวิดีโอที่มีเฟรมเรต (FPS) สูง คุณสามารถกำหนดให้ข้ามเฟรมเพื่อเร่งความเร็วในการประมวลผลได้:
```python
# ประมวลผลทุกๆ เฟรมที่ 5 (ข้าม 4 เฟรม) เพื่อประหยัดการประมวลผล
results = model.predict(source="long_traffic_video.mp4", vid_stride=5)
```
เทคนิคนี้เป็นกลวิธีง่ายๆ แต่มีประสิทธิภาพในทางคอมพิวเตอร์วิทัศน์ เพื่อช่วยรักษาความเร็วการทำงานให้เป็นแบบเรียลไทม์บนอุปกรณ์ที่มีกำลังประมวลผลต่ำ

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX57_Bounding_Box_Parsing_TH]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal|บันทึกการเรียนรู้]]
