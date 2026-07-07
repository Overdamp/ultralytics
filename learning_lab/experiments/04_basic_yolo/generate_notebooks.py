import json
from pathlib import Path

# Structure definitions for the 4 exercises
notebooks_data = {
    "EX61_YOLO_Dataset_Format": {
        "en": {
            "title": "YOLO Dataset Format and Normalized Coordinate Conversion",
            "markdown1": """# 🎓 Bounding Box Coordinate Normalization for YOLO Dataset Format

This notebook explains how bounding box coordinates are converted from raw pixel values to YOLO's normalized coordinate format.

## 1. Concept of YOLO Bounding Box Format
YOLO expects annotations in a `.txt` file corresponding to each image. Each line of the text file contains:
`<class_id> <x_center> <y_center> <width> <height>`

- **`class_id`**: Zero-indexed class ID (integer).
- **`x_center`, `y_center`**: Bounding box center coordinates, normalized by the image's width and height.
- **`width`, `height`**: Bounding box width and height, normalized by the image's width and height.

All coordinates are floats in the range $[0.0, 1.0]$.

## 2. Mathematical Equations
Suppose raw bounding box coordinates are given in $[x_{\\min}, y_{\\min}, x_{\\max}, y_{\\max}]$ (pixel coordinates), and the image has width $W$ and height $H$.

$$\\text{box\\_width} = x_{\\max} - x_{\\min}$$
$$\\text{box\\_height} = y_{\\max} - y_{\\min}$$
$$x_{\\text{center}} = \\frac{x_{\\min} + \\frac{\\text{box\\_width}}{2}}{W} = \\frac{x_{\\min} + x_{\\max}}{2W}$$
$$y_{\\text{center}} = \\frac{y_{\\min} + \\frac{\\text{box\\_height}}{2}}{H} = \\frac{y_{\\min} + y_{\\max}}{2H}$$
$$\\text{normalized\\_width} = \\frac{\\text{box\\_width}}{W}$$
$$\\text{normalized\\_height} = \\frac{\\text{box\\_height}}{H}$$""",
            "code1": """# Verify the EX61_YOLO_Dataset_Format implementation
from solution import convert_to_yolo_format

bbox = [100, 150, 300, 400]
w, h = 640, 480
class_id = 0

result = convert_to_yolo_format(bbox, w, h, class_id)
print(f"Converted YOLO format string: {result}")
assert result == "0 0.312500 0.572917 0.312500 0.520833", "Output coordinates do not match expected!"
print("Verification succeeded!")"""
        },
        "th": {
            "title": "รูปแบบของชุดข้อมูล YOLO และการแปลงพิกัดแบบนอร์มัลไลซ์",
            "markdown1": """# 🎓 การแปลงพิกัดกรอบล้อมรอบแบบนอร์มัลไลซ์สำหรับรูปแบบชุดข้อมูล YOLO

สมุดบันทึกนี้อธิบายวิธีการแปลงพิกัดกรอบล้อมรอบ (Bounding Box) จากพิกัดพิกเซลดิบไปเป็นรูปแบบพิกัดแบบนอร์มัลไลซ์ของ YOLO

## 1. แนวคิดเกี่ยวกับรูปแบบของชุดข้อมูล YOLO
YOLO คาดหวังการระบุป้ายกำกับในไฟล์ `.txt` ที่สอดคล้องกับแต่ละรูปภาพ โดยแต่ละบรรทัดของไฟล์ข้อความจะประกอบด้วย:
`<class_id> <x_center> <y_center> <width> <height>`

- **`class_id`**: รหัสคลาสที่เริ่มต้นด้วยศูนย์ (จำนวนเต็ม)
- **`x_center`, `y_center`**: พิกัดจุดกึ่งกลางของกรอบล้อมรอบ นอร์มัลไลซ์ด้วยความกว้างและความสูงของรูปภาพ
- **`width`, `height`**: ความกว้างและความสูงของกรอบล้อมรอบ นอร์มัลไลซ์ด้วยความกว้างและความสูงของรูปภาพ

พิกัดทั้งหมดเป็นจำนวนทศนิยม (float) ในช่วง $[0.0, 1.0]$

## 2. สมการทางคณิตศาสตร์
สมมติว่าพิกัดกรอบล้อมรอบดิบกำหนดให้อยู่ในรูป $[x_{\\min}, y_{\\min}, x_{\\max}, y_{\\max}]$ (พิกัดพิกเซล) และรูปภาพมีความกว้าง $W$ และความสูง $H$

$$\\text{box\\_width} = x_{\\max} - x_{\\min}$$
$$\\text{box\\_height} = y_{\\max} - y_{\\min}$$
$$x_{\\text{center}} = \\frac{x_{\\min} + \\frac{\\text{box\\_width}}{2}}{W} = \\frac{x_{\\min} + x_{\\max}}{2W}$$
$$y_{\\text{center}} = \\frac{y_{\\min} + \\frac{\\text{box\\_height}}{2}}{H} = \\frac{y_{\\min} + y_{\\max}}{2H}$$
$$\\text{normalized\\_width} = \\frac{\\text{box\\_width}}{W}$$
$$\\text{normalized\\_height} = \\frac{\\text{box\\_height}}{H}$$""",
            "code1": """# Verify the EX61_YOLO_Dataset_Format implementation
from solution import convert_to_yolo_format

bbox = [100, 150, 300, 400]
w, h = 640, 480
class_id = 0

result = convert_to_yolo_format(bbox, w, h, class_id)
print(f"Converted YOLO format string: {result}")
assert result == "0 0.312500 0.572917 0.312500 0.520833", "Output coordinates do not match expected!"
print("Verification succeeded!")"""
        }
    },
    "EX62_Inference_Visualization": {
        "en": {
            "title": "YOLO Inference and Prediction Visualization",
            "markdown1": """# 🎓 YOLO Inference and Prediction Visualization

This notebook demonstrates how to run object detection inference using Ultralytics YOLO, visualize predictions using the built-in `plot()` method, and save the resulting image.

## 1. Concept of Prediction Plotting
When a YOLO model processes an image, it outputs a list of `Results` objects. The `Results` object includes a `plot()` method that:
- Draws bounding boxes around detected objects.
- Places class names and confidence labels next to boxes.
- Returns the annotated image as a BGR numpy array (`numpy.ndarray`).

## 2. Design Decisions
- **Color Space**: `results[0].plot()` returns a BGR image by default to facilitate direct display and saving via OpenCV.
- **File I/O**: We use `cv2.imwrite` to save the BGR image directly to ensure compatibility.
- **Directory Creation**: Before saving, the parent directory of the output path is created programmatically using `pathlib.Path.mkdir(parents=True, exist_ok=True)`.""",
            "code1": """# Verify the EX62_Inference_Visualization implementation
import cv2
import numpy as np
from pathlib import Path
from solution import visualize_and_save

# Create a dummy test image (black square with a white circle)
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(img, (320, 240), 60, (255, 255, 255), -1)
cv2.imwrite("dummy_test.jpg", img)

print("Running inference and visualization...")
# This will download the yolo11n.pt model if not present, run inference, and save
visualize_and_save("yolo11n.pt", "dummy_test.jpg", "output_test.jpg")

output_file = Path("output_test.jpg")
assert output_file.exists(), "Output image was not saved!"
print("Verification succeeded! Output image exists.")

# Clean up
if output_file.exists():
    output_file.unlink()
Path("dummy_test.jpg").unlink()"""
        },
        "th": {
            "title": "การทำนายผลของ YOLO และการแสดงผลลัพธ์พิกัด",
            "markdown1": """# 🎓 การทำนายผลของ YOLO และการแสดงผลลัพธ์พิกัด

สมุดบันทึกนี้แสดงวิธีการเรียกใช้การตรวจจับวัตถุโดยใช้ Ultralytics YOLO, แสดงผลลัพธ์การทำนายโดยใช้เมธอด `plot()` ในตัว และบันทึกรูปภาพผลลัพธ์

## 1. แนวคิดของการพล็อตผลลัพธ์การทำนาย
เมื่อโมเดล YOLO ประมวลผลรูปภาพ จะส่งกลับวัตถุ `Results` ซึ่งเมธอด `plot()` บนวัตถุดังกล่าวจะทำงานดังนี้:
- วาดกรอบล้อมรอบวัตถุที่ตรวจพบ
- วางป้ายกำกับชื่อคลาสและคะแนนความมั่นใจข้างกรอบ
- ส่งคืนภาพที่เขียนคำอธิบายประกอบแล้วในรูปแบบอาเรย์ BGR numpy (`numpy.ndarray`)

## 2. การตัดสินใจในการออกแบบ
- **ปริภูมิสี (Color Space)**: เมธอด `results[0].plot()` จะส่งคืนรูปภาพ BGR เป็นค่าเริ่มต้นเพื่ออำนวยความสะดวกในการแสดงผลและบันทึกโดยตรงผ่าน OpenCV
- **การอ่านเขียนไฟล์ (File I/O)**: เราใช้ `cv2.imwrite` เพื่อบันทึกรูปภาพ BGR โดยตรงเพื่อให้มั่นใจในความเข้ากันได้
- **การสร้างไดเรกทอรี**: ก่อนที่จะบันทึก ไดเรกทอรีหลักของเส้นทางผลลัพธ์จะถูกสร้างขึ้นผ่านโปรแกรมโดยใช้ `pathlib.Path.mkdir(parents=True, exist_ok=True)`""",
            "code1": """# Verify the EX62_Inference_Visualization implementation
import cv2
import numpy as np
from pathlib import Path
from solution import visualize_and_save

# Create a dummy test image (black square with a white circle)
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(img, (320, 240), 60, (255, 255, 255), -1)
cv2.imwrite("dummy_test.jpg", img)

print("Running inference and visualization...")
# This will download the yolo11n.pt model if not present, run inference, and save
visualize_and_save("yolo11n.pt", "dummy_test.jpg", "output_test.jpg")

output_file = Path("output_test.jpg")
assert output_file.exists(), "Output image was not saved!"
print("Verification succeeded! Output image exists.")

# Clean up
if output_file.exists():
    output_file.unlink()
Path("dummy_test.jpg").unlink()"""
        }
    },
    "EX63_Multi_Object_Tracking": {
        "en": {
            "title": "Multi-Object Tracking (MOT) in Video Sequences",
            "markdown1": """# 🎓 Multi-Object Tracking (MOT) in Video Sequences

This notebook covers how to trace object trajectories over consecutive frames using YOLO's tracking API.

## 1. Tracking Concept
While object detection processes each frame independently, tracking links detections across frames using motion dynamics and appearance features to assign unique, persistent track IDs.

## 2. Parameters of `.track()`
- **`tracker`**: The choice of tracking algorithm (e.g., `"bytetrack.yaml"` or `"botsort.yaml"`).
- **`persist=True`**: Essential for maintaining identity associations across frames.
- **`stream=True`**: Memory-efficient generator mode that processes and yields results frame-by-frame.

## 3. Extracting Track IDs
Inside the `Results` object, `results[0].boxes.id` contains the unique tracking ID tensor. Since some frames may have no active tracks, checking for `None` before parsing is necessary.""",
            "code1": """# Verify the EX63_Multi_Object_Tracking implementation
import cv2
import numpy as np
from pathlib import Path
from solution import track_objects_in_video

# Create a dummy video file of 10 frames containing a moving circle
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = "dummy_track.mp4"
out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

for i in range(10):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (100 + i * 20, 240), 40, (255, 255, 255), -1)
    out.write(img)
out.release()

print("Running tracking on synthetic video...")
try:
    track_ids = track_objects_in_video("yolo11n.pt", video_path)
    print(f"Tracking successful! Detected active IDs: {track_ids}")
except Exception as e:
    print(f"Tracking run failed: {e}")
    
# Clean up
Path(video_path).unlink()"""
        },
        "th": {
            "title": "การติดตามวัตถุหลายชิ้น (Multi-Object Tracking) ในลำดับวิดีโอ",
            "markdown1": """# 🎓 การติดตามวัตถุหลายชิ้น (Multi-Object Tracking) ในลำดับวิดีโอ

สมุดบันทึกนี้ครอบคลุมวิธีการติดตามเส้นทางการเคลื่อนที่ของวัตถุข้ามเฟรมต่อเนื่องโดยใช้ API การติดตามของ YOLO

## 1. แนวคิดของการติดตามวัตถุ (Tracking Concept)
ในขณะที่การตรวจจับวัตถุทั่วไปประมวลผลแต่ละเฟรมอย่างเป็นอิสระ การติดตามวัตถุจะเชื่อมโยงผลการตรวจจับข้ามเฟรมต่างๆ โดยใช้การเคลื่อนที่ทางพลศาสตร์และคุณลักษณะของภาพภายนอก เพื่อกำหนดรหัสติดตาม (Track ID) ที่สม่ำเสมอและไม่ซ้ำกัน

## 2. พารามิเตอร์ของ `.track()`
- **`tracker`**: ตัวเลือกอัลกอริทึมการติดตามวัตถุ (เช่น `"bytetrack.yaml"` หรือ `"botsort.yaml"`)
- **`persist=True`**: จำเป็นสำหรับการคงรักษาสถานะและความสัมพันธ์ของวัตถุข้ามเฟรม
- **`stream=True`**: โหมดเจเนอเรเตอร์ประหยัดหน่วยความจำที่จะประมวลผลและส่งคืนผลลัพธ์ทีละเฟรม

## 3. การดึงข้อมูล Track ID
ภายในวัตถุ `Results` ตัวแปร `results[0].boxes.id` จะเก็บเทนเซอร์รหัสติดตามที่ไม่ซ้ำกัน เนื่องจากบางเฟรมอาจไม่มีวัตถุที่กำลังติดตามอยู่ จึงจำเป็นต้องตรวจสอบค่า `None` ก่อนนำข้อมูลไปประมวลผล""",
            "code1": """# Verify the EX63_Multi_Object_Tracking implementation
import cv2
import numpy as np
from pathlib import Path
from solution import track_objects_in_video

# Create a dummy video file of 10 frames containing a moving circle
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = "dummy_track.mp4"
out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

for i in range(10):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (100 + i * 20, 240), 40, (255, 255, 255), -1)
    out.write(img)
out.release()

print("Running tracking on synthetic video...")
try:
    track_ids = track_objects_in_video("yolo11n.pt", video_path)
    print(f"Tracking successful! Detected active IDs: {track_ids}")
except Exception as e:
    print(f"Tracking run failed: {e}")
    
# Clean up
Path(video_path).unlink()"""
        }
    },
    "EX64_Callbacks_and_Logging": {
        "en": {
            "title": "Callback Hooks and Custom Metric Logging in YOLO",
            "markdown1": """# 🎓 Callback Hooks and Custom Metric Logging in YOLO

This notebook demonstrates how to extend YOLO training and validation using custom event callbacks.

## 1. The Callbacks Architecture
Ultralytics YOLO uses a callback system to trigger user-defined functions at key events (e.g., `on_train_start`, `on_val_end`, `on_fit_epoch_end`). This decouples training control and logging logic from core network code.

## 2. Implementing Callback Functions
A callback function registered under validation events (like `on_val_end`) receives the `validator` object as its argument.
- The validation metrics are retrieved using `validator.metrics.results_dict`.
- Bounding box metrics keys are formatted as `"metrics/mAP50(B)"` and `"metrics/mAP50-95(B)"`.""",
            "code1": """# Verify the EX64_Callbacks_and_Logging implementation
from ultralytics import YOLO
from solution import register_custom_metric_callback

print("Initializing YOLO model...")
model = YOLO("yolo11n.pt")

print("Registering custom metric callback...")
model = register_custom_metric_callback(model)

# Verify that callback is registered
callbacks = model.callbacks.get("on_val_end", [])
assert len(callbacks) > 0, "No callback was registered for on_val_end event!"
print("Callback successfully registered in model.callbacks['on_val_end']!")

# Run a quick validation on the small coco8 dataset to trigger the callback
print("Running validation on coco8 dataset...")
try:
    model.val(data="coco8.yaml", imgsz=640, device="cpu", verbose=False)
    print("Callback executed successfully during validation!")
except Exception as e:
    print(f"Validation failed: {e}")"""
        },
        "th": {
            "title": "ระบบคอลแบ็ก (Callback Hooks) และการบันทึกมาตรวัดแบบกำหนดเองใน YOLO",
            "markdown1": """# 🎓 ระบบคอลแบ็ก (Callback Hooks) และการบันทึกมาตรวัดแบบกำหนดเองใน YOLO

สมุดบันทึกนี้แสดงวิธีการขยายขีดความสามารถการฝึกฝนและการตรวจสอบความถูกต้องของ YOLO โดยใช้เหตุการณ์คอลแบ็กแบบกำหนดเอง

## 1. สถาปัตยกรรมคอลแบ็ก (Callbacks Architecture)
Ultralytics YOLO ใช้ระบบคอลแบ็กเพื่อเรียกใช้งานฟังก์ชันที่กำหนดโดยผู้ใช้ในเหตุการณ์สำคัญต่างๆ (เช่น `on_train_start`, `on_val_end`, `on_fit_epoch_end`) สิ่งนี้ช่วยแยกตรรกะการควบคุมและการบันทึกข้อมูลออกจากการทำงานของโครงข่ายประสาทหลัก

## 2. การสร้างฟังก์ชันคอลแบ็ก
ฟังก์ชันคอลแบ็กที่ลงทะเบียนในเหตุการณ์การวัดผล (เช่น `on_val_end`) จะรับวัตถุ `validator` เป็นอาร์กิวเมนต์
- มาตรวัดความถูกต้องจะถูกดึงออกมาโดยใช้ `validator.metrics.results_dict`
- คีย์สำหรับมาตรวัดระดับกรอบล้อมรอบ (Bounding Box) จะอยู่ในรูปแบบ `"metrics/mAP50(B)"` และ `"metrics/mAP50-95(B)"`""",
            "code1": """# Verify the EX64_Callbacks_and_Logging implementation
from ultralytics import YOLO
from solution import register_custom_metric_callback

print("Initializing YOLO model...")
model = YOLO("yolo11n.pt")

print("Registering custom metric callback...")
model = register_custom_metric_callback(model)

# Verify that callback is registered
callbacks = model.callbacks.get("on_val_end", [])
assert len(callbacks) > 0, "No callback was registered for on_val_end event!"
print("Callback successfully registered in model.callbacks['on_val_end']!")

# Run a quick validation on the small coco8 dataset to trigger the callback
print("Running validation on coco8 dataset...")
try:
    model.val(data="coco8.yaml", imgsz=640, device="cpu", verbose=False)
    print("Callback executed successfully during validation!")
except Exception as e:
    print(f"Validation failed: {e}")"""
        }
    }
}

base_dir = Path("/home/luke/ai_training/ultralytics/learning_lab/experiments/04_basic_yolo")

for ex_name, ex_data in notebooks_data.items():
    ex_path = base_dir / ex_name
    ex_path.mkdir(parents=True, exist_ok=True)
    
    # Write English notebook
    en_nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ex_data["en"]["markdown1"].splitlines(keepends=True)
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ex_data["en"]["code1"].splitlines(keepends=True)
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Write Thai notebook
    th_nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ex_data["th"]["markdown1"].splitlines(keepends=True)
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ex_data["th"]["code1"].splitlines(keepends=True)
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(ex_path / "explanation.ipynb", "w", encoding="utf-8") as f:
        json.dump(en_nb, f, indent=2, ensure_ascii=False)
        
    with open(ex_path / "explanation_TH.ipynb", "w", encoding="utf-8") as f:
        json.dump(th_nb, f, indent=2, ensure_ascii=False)

print("All notebooks generated successfully.")
