# 🧠 EX55: โหมดการประเมินผลและตัวชี้วัดการตรวจจับวัตถุ (Evaluation Modes & Detection Metrics)

การประเมินผลโมเดลคือกระบวนการแปลงผลการทำนายให้กลายเป็นตัวชี้วัดเชิงปริมาณที่ตอบว่า "โมเดลดีแค่ไหนในงานจริง?" บทเรียนนี้จะอธิบายทั้งคณิตศาสตร์เบื้องหลัง mAP และวิธีดึงผลการประเมินผ่าน Python API

---

## 1. รากฐานทางคณิตศาสตร์ของการประเมินผล

### A. Intersection over Union (IoU)

IoU วัดว่ากล่องที่โมเดลทำนาย ($\hat{B}$) ซ้อนทับกับกล่องเฉลยจริง ($B_{gt}$) มากน้อยเพียงใด:

$$\text{IoU}(\hat{B}, B_{gt}) = \frac{|\hat{B} \cap B_{gt}|}{|\hat{B} \cup B_{gt}|}$$

*   $\text{IoU} = 1.0$ → ซ้อนทับกันสมบูรณ์แบบ
*   $\text{IoU} = 0.0$ → ไม่มีการซ้อนทับใดๆ
*   ค่า threshold มาตรฐาน: $\text{IoU} \geq 0.5$ → ถือว่าตรวจจับถูก (True Positive)

### B. Precision และ Recall

สำหรับคลาสหนึ่งๆ ณ ค่า confidence threshold และ IoU threshold ที่กำหนด:

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{ตรวจจับถูกต้อง}}{\text{ที่โมเดลรายงานว่าพบทั้งหมด}}$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{ตรวจจับถูกต้อง}}{\text{ที่มีอยู่จริงทั้งหมดในชุดข้อมูล}}$$

โดยที่:
*   **TP** (True Positive): ทำนายว่าพบวัตถุ และ IoU ≥ threshold → ถูกต้อง
*   **FP** (False Positive): ทำนายว่าพบวัตถุ แต่ IoU < threshold → ผิดพลาด (ตรวจจับเกินจริง)
*   **FN** (False Negative): ไม่ได้ทำนายว่าพบ ทั้งที่มีวัตถุอยู่จริง → ตกหล่น

### C. Average Precision (AP) — พื้นที่ใต้ Precision-Recall Curve

AP คำนวณโดยการเรียง detection ทั้งหมดตาม confidence score จากสูงไปต่ำ แล้วประเมิน Precision และ Recall สะสม แล้วคำนวณพื้นที่ใต้เส้นกราฟ (AUC):

$$\text{AP} = \int_0^1 P(R)\,dR \approx \sum_{k=1}^{N} (R_k - R_{k-1})\,P_k$$

*ในทางปฏิบัติ YOLO ใช้การประมาณ AP แบบ 101-point interpolation ตามมาตรฐาน COCO*

### D. mAP@0.5 vs mAP@0.5:0.95

| ตัวชี้วัด | สูตรคำนวณ | ความหมาย |
| :--- | :--- | :--- |
| **mAP@0.5** | $\frac{1}{C}\sum_{c=1}^{C}\text{AP}_c^{IoU=0.5}$ | เฉลี่ย AP ทุกคลาส ณ IoU threshold = 0.5 (standard recall) |
| **mAP@0.5:0.95** | $\frac{1}{C \cdot 10}\sum_{c=1}^{C}\sum_{t=0.5}^{0.95}\text{AP}_c^{IoU=t}$ | เฉลี่ย AP ทุกคลาสและทุก IoU threshold (0.50, 0.55, ..., 0.95) — มาตรฐาน COCO |

**ข้อแตกต่างสำคัญ:** โมเดลที่มี mAP@0.5 สูงแต่ mAP@0.5:0.95 ต่ำ หมายความว่าโมเดลหาวัตถุเจอ แต่ลากกล่องขอบเขตได้หลวม ไม่แม่นยำ

---

## 2. ความแตกต่างระหว่าง Validation และ Inference

| คุณสมบัติ | `model.val()` | `model.predict()` |
| :--- | :--- | :--- |
| **ต้องการป้ายกำกับ** | ✅ ใช่ (อ่านจาก `data.yaml`) | ❌ ไม่ต้องการ |
| **ผลลัพธ์ที่ได้** | Precision, Recall, mAP, Confusion Matrix | พิกัดกล่อง, ป้ายคลาส, confidence score |
| **วัตถุประสงค์** | วัดประสิทธิภาพโมเดล | นำโมเดลไปใช้งานจริง |
| **ข้อมูล ground-truth** | เปรียบเทียบกับ label ที่มีอยู่ | ไม่มีการเปรียบเทียบ |

---

## 3. การดึงตัวชี้วัดด้วย Python API

```python
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกสอนสำเร็จแล้ว (ใช้ best.pt สำหรับการประเมินผลจริง)
model = YOLO("runs/detect/train/weights/best.pt")

# รัน Validation บนชุดข้อมูล val ที่ระบุใน data.yaml
# ค่า conf=0.001 ต่ำมากเพื่อดึง detection ทั้งหมดมาสร้าง PR Curve ที่สมบูรณ์
metrics = model.val(data="datasets/custom_data/data.yaml", conf=0.001, iou=0.6)

# ─── ตัวชี้วัดภาพรวม ───────────────────────────────────────────
print("=== ผลการประเมินภาพรวม ===")
print(f"  Precision (P):   {metrics.box.mp:.4f}")
print(f"  Recall    (R):   {metrics.box.mr:.4f}")
print(f"  mAP@0.5:         {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95:    {metrics.box.map:.4f}")

# ─── ตัวชี้วัดแยกรายคลาส ────────────────────────────────────────
print("\n=== ผลการประเมินแยกรายคลาส ===")
class_names = metrics.names  # {0: 'valve', 1: 'pipe', ...}
for class_idx, class_name in class_names.items():
    # class_result(i) ส่งคืน tuple: (Precision, Recall, AP50, AP50-95)
    p, r, ap50, ap = metrics.box.class_result(class_idx)
    print(f"  [{class_idx}] {class_name:<20} P={p:.3f}  R={r:.3f}  AP50={ap50:.3f}  AP={ap:.3f}")

# ─── ตรวจสอบ Confusion Matrix ──────────────────────────────────
print(f"\nConfusion Matrix บันทึกที่: {metrics.save_dir}")
```

### CLI Equivalent
```bash
yolo val model=runs/detect/train/weights/best.pt \
          data=datasets/custom_data/data.yaml \
          conf=0.001 iou=0.6
```

---

## 4. การประเมินผลบน Test Split

> [!IMPORTANT]
> ตรวจสอบว่า `data.yaml` ของคุณมีส่วน `test:` ก่อนรัน test evaluation เสมอ:
> ```bash
> # ตรวจสอบโครงสร้าง data.yaml
> ```
> 
> ```python
> import yaml
> with open("datasets/custom_data/data.yaml") as f:
>     cfg = yaml.safe_load(f)
> print(cfg.get("test", "ไม่พบ test split!"))
> ```

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# ประเมินบน test set (ไม่ใช่ val set)
test_metrics = model.val(
    data="datasets/custom_data/data.yaml",
    split="test",   # ระบุ split ที่ต้องการ
    save_json=True  # บันทึกผลในรูปแบบ COCO JSON สำหรับวิเคราะห์เพิ่มเติม
)
print(f"Test mAP@0.5:0.95 = {test_metrics.box.map:.4f}")
```

---

## 💡 คำแนะนำจากอาจารย์: การวิเคราะห์ Precision-Recall Trade-off

**การปรับ `conf` threshold มีผลโดยตรงต่อ P/R:**

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# ทดลองค่า confidence threshold ต่างๆ
for conf_thresh in [0.1, 0.25, 0.5, 0.7]:
    m = model.val(data="custom.yaml", conf=conf_thresh, verbose=False)
    print(f"conf={conf_thresh:.2f} → P={m.box.mp:.3f}, R={m.box.mr:.3f}, mAP50={m.box.map50:.3f}")
```

**หลักการตีความ:**

| สถานการณ์ | ความหมาย | กลยุทธ์แก้ไข |
| :--- | :--- | :--- |
| P สูง, R ต่ำ | โมเดลระมัดระวัง — แน่ใจก่อนรายงาน แต่ตกหล่นมาก | ลด `conf` threshold ลง |
| P ต่ำ, R สูง | โมเดลดุดัน — รายงานครบแต่มี false positive มาก | เพิ่ม `conf` threshold |
| mAP50 สูง, mAP ต่ำ | ตรวจพบวัตถุ แต่ลากกล่องหลวม | เพิ่มข้อมูล annotation ที่แม่นยำขึ้น |

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX54_Resuming_Training_TH|EX54: การฝึกสอนต่อจากเดิม (Resuming Training)]]
*   [[EX56_Prediction_Sources_TH|EX56: แหล่งข้อมูลสำหรับการทำนาย (Prediction Sources)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal|บันทึกการเรียนรู้]]
