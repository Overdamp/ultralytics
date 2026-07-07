# 🧠 EX54: การฝึกสอนต่อจากจุดที่ถูกขัดจังหวะ (Resuming Interrupted Training)

การฝึกสอนโมเดล Deep Learning อาจใช้เวลาหลายชั่วโมงถึงหลายวัน ข้อขัดข้องที่เกิดขึ้นกลางคัน (ไฟดับ, OOM, Cloud Preemption) เป็นเรื่องปกติ Ultralytics YOLO มีระบบ **Checkpointing** ในตัวที่ช่วยให้กู้คืนสถานะการฝึกสอนได้อย่างสมบูรณ์โดยไม่สูญเสียความก้าวหน้า

---

## 1. รากฐานทางคณิตศาสตร์: อะไรอยู่ภายใน `last.pt`?

ไฟล์จุดตรวจสอบ `last.pt` ไม่ใช่แค่ค่าน้ำหนักโมเดล — มันคือ **State Dictionary** ที่บรรจุสถานะเต็มรูปแบบของกระบวนการออพติไมซ์ ทำให้การฝึกสอนต่อเหมือนไม่เคยหยุดเลย:

```python
# โครงสร้างภายในของ last.pt (พจนานุกรม PyTorch checkpoint)
checkpoint = {
    'epoch':       42,           # รอบล่าสุดที่เสร็จสมบูรณ์ (ดัชนีเริ่มจาก 0)
    'best_fitness': 0.712,       # คะแนน fitness ดีที่สุดจนถึงปัจจุบัน (สำหรับบันทึก best.pt)
    'model':        ...,         # state_dict ของโครงข่ายประสาทเทียม (ค่าน้ำหนักทั้งหมด)
    'ema':          ...,         # state_dict ของ EMA model (Exponential Moving Average)
    'optimizer':    ...,         # สถานะโมเมนตัมและ moment estimates ของ optimizer
    'train_args':   {...},       # อาร์กิวเมนต์ดั้งเดิม (data, epochs, batch, imgsz ฯลฯ)
    'date':         '2026-07-07',
}
```

### A. ทำไมต้องบันทึก Optimizer State?

ตัวออพติไมเซอร์ AdamW (ที่ YOLO ใช้เป็นค่าเริ่มต้น) ติดตาม **moment estimates** สำหรับทุกพารามิเตอร์:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t \quad \text{(Momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2 \quad \text{(Velocity)}$$

หากเราโหลดแต่ค่าน้ำหนัก $\theta$ โดยไม่โหลด $m_t$ และ $v_t$ กลับมา ตัวออพติไมเซอร์จะ**เริ่มนับใหม่จากศูนย์** ทำให้อัตราการเรียนรู้ที่ปรับตัวได้ (adaptive learning rate) ไม่สอดคล้องกับสถานะของโมเดล ณ ขณะนั้น ส่งผลให้โมเดลทำงานแย่ลงชั่วคราวหรือไม่ลู่เข้าเลยในรอบแรกๆ

### B. ทำไมต้องบันทึก LR Scheduler State?

YOLO ใช้ Cosine Annealing ลดอัตราการเรียนรู้ตามสมการ:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{T_{cur}}{T_{max}}\pi\right)$$

Scheduler State บันทึก $T_{cur}$ (รอบปัจจุบัน) ไว้ด้วย ดังนั้นเมื่อ resume โมเดลจะใช้ $\eta$ ที่ถูกต้องสำหรับรอบที่ 43 (ต่อจากที่หยุดไว้) ไม่ใช่รีเซ็ตกลับไปที่ `lr0` เริ่มต้น

### C. ทำไมต้องบันทึก EMA?

`ema` คือ Exponential Moving Average ของค่าน้ำหนัก ซึ่งเป็นค่าที่ YOLO ใช้สร้าง `best.pt` สำหรับการทำนายผลจริง:

$$\theta_{EMA,t} = \alpha\,\theta_{EMA,t-1} + (1-\alpha)\,\theta_t \quad (\alpha \approx 0.9999)$$

---

## 2. การฝึกสอนต่อด้วย Python API

```python
from ultralytics import YOLO

# โหลดจุดตรวจสอบรอบล่าสุด — ต้องเป็น last.pt เสมอ ไม่ใช่ best.pt
model = YOLO("runs/detect/train/weights/last.pt")

# resume=True จะสั่งให้อ่านค่าคอนฟิกทั้งหมดจากไฟล์ checkpoint
# ห้ามส่ง epochs, data, batch หรือพารามิเตอร์อื่นเพิ่มเติม เพราะจะทับค่าเดิม
results = model.train(resume=True)

print(f"ฝึกสอนเสร็จสมบูรณ์ บันทึกผลลัพธ์ที่: {results.save_dir}")
```

> [!IMPORTANT]
> **ทำไม `last.pt` ไม่ใช่ `best.pt`?**
> `best.pt` บันทึกเฉพาะค่าน้ำหนัก EMA ที่ดีที่สุด *ณ ขณะนั้น* แต่ไม่มี optimizer state หรือ scheduler state จึง **ไม่สามารถ resume ได้อย่างถูกต้อง**

### CLI Equivalent
```bash
yolo train resume model=runs/detect/train/weights/last.pt
```

---

## 3. การตรวจสอบสถานะ Checkpoint ด้วย Python (Programmatic Inspection)

ก่อน resume เสมอ ควรตรวจสอบว่า checkpoint มีข้อมูลครบถ้วนหรือไม่:

```python
import torch

checkpoint_path = "runs/detect/train/weights/last.pt"
ckpt = torch.load(checkpoint_path, map_location="cpu")

# ตรวจสอบรอบที่หยุดค้างไว้
print(f"หยุดอยู่ที่รอบ (epoch): {ckpt.get('epoch', 'ไม่มีข้อมูล')}")
print(f"คะแนน Fitness ดีที่สุดจนถึงปัจจุบัน: {ckpt.get('best_fitness', 'ไม่มีข้อมูล'):.4f}")

# ตรวจสอบว่า optimizer state ยังอยู่ครบ
has_optimizer = 'optimizer' in ckpt and ckpt['optimizer'] is not None
print(f"มี Optimizer State: {has_optimizer}")

# อ่านอาร์กิวเมนต์การเทรนดั้งเดิม
train_args = ckpt.get('train_args', {})
print(f"ชุดข้อมูลเดิม: {train_args.get('data', 'ไม่ทราบ')}")
print(f"จำนวนรอบทั้งหมดเดิม: {train_args.get('epochs', 'ไม่ทราบ')}")
```

---

## 4. การแก้ไขพารามิเตอร์เมื่อ Resume หลัง OOM

หากการฝึกสอนหยุดเพราะหน่วยความจำ GPU เต็ม (OOM) และต้องการลด `batch` ก่อน resume:

```python
import yaml
from pathlib import Path

# ค้นหาและแก้ไขไฟล์ args.yaml ของรันที่ถูกขัดจังหวะ
args_path = Path("runs/detect/train/args.yaml")

with open(args_path, "r") as f:
    args = yaml.safe_load(f)

print(f"ขนาด batch เดิม: {args['batch']}")

# ลดขนาด batch ลงครึ่งหนึ่ง
args['batch'] = args['batch'] // 2
print(f"ขนาด batch ใหม่: {args['batch']}")

with open(args_path, "w") as f:
    yaml.dump(args, f, default_flow_style=False, allow_unicode=True)

print("บันทึกการแก้ไขเรียบร้อย กำลัง resume...")

from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True)
```

---

## 💡 คำแนะนำจากอาจารย์ (Professor Tips)

### สาเหตุที่พบบ่อยของการล้มเหลวใน Resume

| สาเหตุ | อาการ | วิธีแก้ไข |
| :--- | :--- | :--- |
| โหลด `best.pt` แทน `last.pt` | ข้อผิดพลาด หรือโมเดลเริ่มต้นใหม่ | ใช้ `last.pt` เสมอ |
| ส่ง `epochs=N` ใหม่เข้าไป | YOLO รันเพิ่มอีก N รอบจากจุดที่หยุด | ใช้ `resume=True` โดยไม่ส่งพารามิเตอร์อื่น |
| `data.yaml` ถูกเคลื่อนย้าย | ข้อผิดพลาด FileNotFound | แก้ไข path ใน `args.yaml` ก่อน resume |
| Checkpoint เสียหาย (disk error) | ข้อผิดพลาดในการโหลด | ตรวจสอบ integrity ด้วย `torch.load()` ก่อน |

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX53_Training_Settings_TH|EX53: การตั้งค่าการฝึกสอน (Training Settings)]]
*   [[EX55_Evaluation_Modes_TH|EX55: โหมดการประเมินผล (Evaluation Modes)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal|บันทึกการเรียนรู้]]
