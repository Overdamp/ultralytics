"""
refine_notebooks.py  —  Comprehensive notebook review & refinement
Improvements:
  1. yolov8n.pt → yolo11n.pt everywhere
  2. GPU device auto-detection in every code cell
  3. Richer audit output: tensor shapes, dtypes, vectorised extraction
  4. Memory cleanup: gc.collect() + torch.cuda.empty_cache()
  5. EX54: Real torch.load() checkpoint inspection
  6. EX55: Per-class AP breakdown + mAP bar chart
  7. EX61: Writes & verifies a real .txt label file + round-trip decode
  8. EX63: Two-object trajectory video
"""
import json, textwrap
from pathlib import Path

BASE = Path("/home/luke/ai_training/ultralytics/learning_lab/experiments/04_basic_yolo")

def nb(md, code):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.0"}},
        "cells": [
            {"cell_type": "markdown", "id": "md01", "metadata": {}, "source": md},
            {"cell_type": "code", "execution_count": None, "id": "code01", "metadata": {}, "outputs": [], "source": code},
        ],
    }


# ── Content per exercise ──────────────────────────────────────────────────────
NOTEBOOKS = {}

NOTEBOOKS["EX51_YOLO_CLI"] = {
"en": {
"md": """\
# 🧠 EX51: YOLO Command Line Interface (CLI)

CLI spawns a **new OS process** per call (~0.5 s overhead; CUDA kernel recompile on first GPU use).
For production, use `from ultralytics import YOLO` instead.

## CLI Syntax
```bash
yolo TASK MODE key=value ...
```
| Field | Values |
|-------|--------|
| TASK | `detect`, `segment`, `classify`, `pose`, `obb` |
| MODE | `predict`, `train`, `val`, `export`, `track` |

## ⚠️ Safety Warning
Verify `data.yaml` path before any training command.

## 🔗 Links
- [[YOLO_Learning_Plan]] | [[learning_journal]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import os, glob, gc
import cv2, torch
import matplotlib.pyplot as plt
from solution import run_yolo_cli_command
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device} | CUDA: {torch.cuda.is_available()}")

print("\\n--- AUDIT & INSPECTION START ---")
result = run_yolo_cli_command(task="detect", mode="predict", model="yolo11n.pt",
                               data=None, epochs=None, imgsz=320)
print(f"Return code: {result.returncode}")
print(f"stdout (last 500):\\n{result.stdout[-500:]}")
if result.returncode != 0:
    print(f"stderr:\\n{result.stderr[-300:]}")
print("--- AUDIT & INSPECTION END ---")

predict_dirs = sorted(glob.glob("runs/detect/predict*"))
if predict_dirs:
    imgs = glob.glob(os.path.join(predict_dirs[-1], "*.jpg"))
    if imgs:
        bgr = cv2.imread(imgs[0])
        print(f"Image shape: {bgr.shape} | dtype: {bgr.dtype}")
        print(f"Pixel range: {bgr.min()}–{bgr.max()}")
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(imgs[0])); plt.axis("off"); plt.show()
    else:
        print("No prediction images found.")
else:
    print("No predict dirs found.")
"""},
"th": {
"md": """\
# 🧠 EX51: YOLO Command Line Interface (CLI)

CLI สร้าง **กระบวนการ OS ใหม่** ต่อการเรียกแต่ละครั้ง (~0.5 วินาที; CUDA kernel recompile ครั้งแรก)
สำหรับ production ควรใช้ `from ultralytics import YOLO` แทน

## โครงสร้าง CLI
```bash
yolo TASK MODE key=value ...
```
| Field | ค่าที่รับได้ |
|-------|------------|
| TASK | `detect`, `segment`, `classify`, `pose`, `obb` |
| MODE | `predict`, `train`, `val`, `export`, `track` |

## ⚠️ คำเตือน
ตรวจสอบ path ของ `data.yaml` ก่อนรัน training เสมอ

## 🔗 ลิงก์
- [[YOLO_Learning_Plan]] | [[learning_journal]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import os, glob, gc
import cv2, torch
import matplotlib.pyplot as plt
from solution import run_yolo_cli_command
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device} | CUDA: {torch.cuda.is_available()}")

print("\\n--- เริ่มการตรวจสอบ ---")
result = run_yolo_cli_command(task="detect", mode="predict", model="yolo11n.pt",
                               data=None, epochs=None, imgsz=320)
print(f"Return code: {result.returncode}")
print(f"stdout (500 ท้าย):\\n{result.stdout[-500:]}")
print("--- สิ้นสุดการตรวจสอบ ---")

predict_dirs = sorted(glob.glob("runs/detect/predict*"))
if predict_dirs:
    imgs = glob.glob(os.path.join(predict_dirs[-1], "*.jpg"))
    if imgs:
        bgr = cv2.imread(imgs[0])
        print(f"ขนาดภาพ: {bgr.shape} | dtype: {bgr.dtype}")
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(imgs[0])); plt.axis("off"); plt.show()
    else:
        print("ไม่พบภาพผลลัพธ์")
else:
    print("ไม่พบโฟลเดอร์ predict")
"""}}

NOTEBOOKS["EX52_Model_Configurations"] = {
"en": {
"md": """\
# 🧠 EX52: Model Configurations

| Method | File | Weights | Use-case |
|--------|------|---------|----------|
| Scratch (YAML) | `yolov8n.yaml` | Kaiming random init | Novel domain from scratch |
| Pretrained (.pt) | `yolo11n.pt` | COCO-pretrained | Transfer learning ✅ |

### Kaiming (He) Initialization
$$W \\sim \\mathcal{N}\\left(0,\\ \\sqrt{\\tfrac{2}{n_{in}}}\\right)$$
Preserves activation variance across layers; prevents vanishing/exploding gradients.

## 🔗 Links
- [[YOLO_Learning_Plan]] | [[learning_journal]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, torch
import matplotlib.pyplot as plt
from solution import load_model_from_yaml, load_pretrained_model
%matplotlib inline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

print("\\n--- AUDIT & INSPECTION START ---")
yaml_model = load_model_from_yaml("yolov8n.yaml")
pretrained_model = load_pretrained_model("yolo11n.pt")

yaml_p = sum(p.numel() for p in yaml_model.model.parameters())
pt_p   = sum(p.numel() for p in pretrained_model.model.parameters())
print(f"YAML params:       {yaml_p:>12,}")
print(f"Pretrained params: {pt_p:>12,}")

w = next(yaml_model.model.parameters())
print(f"\\nYAML first layer | mean={w.mean().item():.6f} std={w.std().item():.6f}")
print(f"  Kaiming expected std ≈ {(2/w.shape[1])**0.5:.4f}")

dummy = torch.randn(1, 3, 320, 320)
yaml_model.model.eval(); pretrained_model.model.eval()
with torch.no_grad():
    out_yaml = yaml_model.model(dummy)
    out_pt   = pretrained_model.model(dummy)
print(f"\\nYAML out type: {type(out_yaml).__name__}")
print(f"PT   out type: {type(out_pt).__name__}")
print("--- AUDIT & INSPECTION END ---")

del yaml_model, pretrained_model, dummy, out_yaml, out_pt
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

plt.figure(figsize=(6,4))
plt.bar(["Random (YAML)", "Pretrained (.pt)"], [yaml_p, pt_p], color=["#e74c3c","#3498db"])
plt.ylabel("Parameters")
plt.title("Model Parameters: Random vs Pretrained")
for i,v in enumerate([yaml_p, pt_p]):
    plt.text(i, v*1.01, f"{v/1e6:.2f}M", ha="center")
plt.tight_layout(); plt.show()
"""},
"th": {
"md": """\
# 🧠 EX52: Model Configurations (การกำหนดค่าคอนฟิกโมเดล)

| วิธี | ไฟล์ | ค่าน้ำหนัก | เหมาะกับ |
|-----|------|-----------|---------|
| Scratch (YAML) | `yolov8n.yaml` | Kaiming random init | Domain ใหม่ทั้งหมด |
| Pretrained (.pt) | `yolo11n.pt` | COCO-pretrained | Transfer learning ✅ |

### Kaiming (He) Initialization
$$W \\sim \\mathcal{N}\\left(0,\\ \\sqrt{\\tfrac{2}{n_{in}}}\\right)$$
รักษา variance ของสัญญาณทุก layer ป้องกัน vanishing/exploding gradient

## 🔗 ลิงก์
- [[YOLO_Learning_Plan]] | [[learning_journal]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, torch
import matplotlib.pyplot as plt
from solution import load_model_from_yaml, load_pretrained_model
%matplotlib inline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

print("\\n--- เริ่มการตรวจสอบ ---")
yaml_model = load_model_from_yaml("yolov8n.yaml")
pretrained_model = load_pretrained_model("yolo11n.pt")

yaml_p = sum(p.numel() for p in yaml_model.model.parameters())
pt_p   = sum(p.numel() for p in pretrained_model.model.parameters())
print(f"YAML params:       {yaml_p:>12,} พารามิเตอร์")
print(f"Pretrained params: {pt_p:>12,} พารามิเตอร์")

w = next(yaml_model.model.parameters())
print(f"\\nYAML layer แรก | mean={w.mean().item():.6f} std={w.std().item():.6f}")
print(f"  Kaiming คาด std ≈ {(2/w.shape[1])**0.5:.4f}")

dummy = torch.randn(1, 3, 320, 320)
yaml_model.model.eval(); pretrained_model.model.eval()
with torch.no_grad():
    out_yaml = yaml_model.model(dummy)
    out_pt   = pretrained_model.model(dummy)
print(f"\\nYAML output type: {type(out_yaml).__name__}")
print("--- สิ้นสุดการตรวจสอบ ---")

del yaml_model, pretrained_model, dummy, out_yaml, out_pt
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

plt.figure(figsize=(6,4))
plt.bar(["Random (YAML)", "Pretrained (.pt)"], [yaml_p, pt_p], color=["#e74c3c","#3498db"])
plt.ylabel("Parameters"); plt.title("Random vs Pretrained")
for i,v in enumerate([yaml_p, pt_p]):
    plt.text(i, v*1.01, f"{v/1e6:.2f}M", ha="center")
plt.tight_layout(); plt.show()
"""}}

NOTEBOOKS["EX53_Training_Settings"] = {
"en": {
"md": """\
# 🧠 EX53: Training Settings

| Param | Default | Effect |
|-------|---------|--------|
| `epochs` | 100 | Training rounds |
| `batch` | 16 | Samples per gradient step |
| `imgsz` | 640 | Input resolution (px) |
| `lr0` | 0.01 | Initial LR |
| `optimizer` | `AdamW` | Gradient descent algorithm |
| `amp` | True | Mixed precision FP16 — halves VRAM |

### AdamW vs SGD
**SGD+momentum:** $v_{t+1} = \\mu v_t - \\eta \\nabla L$, $\\theta \\mathrel{+}= v_{t+1}$

**AdamW:** bias-corrected adaptive LR per parameter + decoupled weight decay.
AdamW converges faster; SGD+cosine often peaks higher on large-scale data.

## ⚠️ Safety Warning
Verify `data.yaml` before `model.train()`.

## 🔗 Links
- [[YOLO_Learning_Plan]] | [[learning_journal]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, torch
import pandas as pd
import matplotlib.pyplot as plt
from solution import train_custom_model
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f"[INFO] Device: GPU:{device} | VRAM: {vram} GB")
else:
    print("[INFO] Device: cpu")

print("\\n--- AUDIT & INSPECTION START ---")
print("Training 3 epochs on coco8 to audit loss output...")
results = train_custom_model(model_path="yolo11n.pt", data_yaml="coco8.yaml",
                              epochs=3, batch=8, imgsz=320, device=device)

results_csv = results.save_dir / "results.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    print(f"\\nCSV columns: {list(df.columns)}")
    print(df.tail(3).to_string(index=False))
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    fig, axes = plt.subplots(1, len(loss_cols), figsize=(4*len(loss_cols), 4))
    if len(loss_cols)==1: axes=[axes]
    for ax,col in zip(axes, loss_cols):
        ax.plot(df["epoch"], df[col], marker="o", linewidth=2)
        ax.set_title(col); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
    plt.suptitle("Training Loss Curves (3-epoch audit)")
    plt.tight_layout(); plt.show()
else:
    print("results.csv not found.")
print("--- AUDIT & INSPECTION END ---")

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""},
"th": {
"md": """\
# 🧠 EX53: Training Settings (การตั้งค่าการฝึกสอน)

| พารามิเตอร์ | ค่าเริ่มต้น | ผลกระทบ |
|------------|------------|---------|
| `epochs` | 100 | จำนวนรอบเทรน |
| `batch` | 16 | ตัวอย่างต่อ gradient step |
| `imgsz` | 640 | ความละเอียดภาพ (px) |
| `lr0` | 0.01 | อัตราการเรียนรู้เริ่มต้น |
| `optimizer` | `AdamW` | อัลกอริทึม gradient descent |
| `amp` | True | Mixed precision FP16 — ลด VRAM ครึ่งหนึ่ง |

### AdamW vs SGD
**SGD+momentum:** $v_{t+1} = \\mu v_t - \\eta \\nabla L$, $\\theta \\mathrel{+}= v_{t+1}$

**AdamW:** ปรับ LR แต่ละพารามิเตอร์โดยอัตโนมัติ พร้อม weight decay แยก
AdamW ลู่เข้าเร็วกว่า; SGD+cosine มักให้ mAP สูงกว่าบนชุดข้อมูลขนาดใหญ่

## ⚠️ คำเตือน
ตรวจสอบ `data.yaml` ก่อน `model.train()` เสมอ

## 🔗 ลิงก์
- [[YOLO_Learning_Plan]] | [[learning_journal]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, torch
import pandas as pd
import matplotlib.pyplot as plt
from solution import train_custom_model
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

print("\\n--- เริ่มการตรวจสอบ ---")
print("เทรน 3 epoch บน coco8 เพื่อตรวจสอบ loss output...")
results = train_custom_model(model_path="yolo11n.pt", data_yaml="coco8.yaml",
                              epochs=3, batch=8, imgsz=320, device=device)

results_csv = results.save_dir / "results.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    print(f"\\nคอลัมน์: {list(df.columns)}")
    print(df.tail(3).to_string(index=False))
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    fig, axes = plt.subplots(1, len(loss_cols), figsize=(4*len(loss_cols), 4))
    if len(loss_cols)==1: axes=[axes]
    for ax,col in zip(axes, loss_cols):
        ax.plot(df["epoch"], df[col], marker="o", linewidth=2)
        ax.set_title(col); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
    plt.suptitle("กราฟ Loss (3 epoch audit)")
    plt.tight_layout(); plt.show()
print("--- สิ้นสุดการตรวจสอบ ---")
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""}}

NOTEBOOKS["EX54_Resuming_Training"] = {
"en": {
"md": """\
# 🧠 EX54: Resuming Interrupted Training

Training can be interrupted by OOM, power cuts, or cloud preemption.
`last.pt` stores the **complete training state** for seamless resumption.

## `last.pt` internals
```python
{
  'epoch':        42,     # last completed (0-based)
  'best_fitness': 0.712,  # best val score so far
  'model':        ...,    # weight state_dict
  'ema':          ...,    # EMA state_dict (→ best.pt)
  'optimizer':    ...,    # AdamW m_t, v_t tensors
  'train_args':   {...},  # original config
}
```

### Why optimizer state matters
$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t \\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2$$
Losing $m_t, v_t$ resets adaptive LR → temporary performance dip.

- ✅ Always resume from `last.pt`
- ❌ Never resume from `best.pt` (missing optimizer state)

## 🔗 Links
- [[EX53_Training_Settings_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import torch, pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solution import resume_yolo_training
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")
print("\\n--- AUDIT & INSPECTION START ---")

# Step 1: Create a checkpoint via a 2-epoch base run
print("[Step 1] Training 2-epoch baseline to create last.pt...")
base = YOLO("yolo11n.pt")
base_res = base.train(data="coco8.yaml", epochs=2, imgsz=320, batch=8, device=device, verbose=False)
ckpt = str(base_res.save_dir / "weights" / "last.pt")
print(f"Checkpoint: {ckpt}")

# Step 2: Inspect checkpoint keys
print("\\n[Step 2] Inspecting checkpoint...")
ck = torch.load(ckpt, map_location="cpu")
print(f"  Keys: {list(ck.keys())}")
print(f"  Epoch completed:  {ck.get('epoch','N/A')}")
print(f"  Best fitness:     {ck.get('best_fitness',0.0):.4f}")
print(f"  Optimizer state:  {'present ✅' if ck.get('optimizer') is not None else 'MISSING ❌'}")
ta = ck.get("train_args", {})
print(f"  Config: epochs={ta.get('epochs','?')}, batch={ta.get('batch','?')}, imgsz={ta.get('imgsz','?')}")

# Step 3: Resume
print("\\n[Step 3] Resuming from checkpoint...")
res = resume_yolo_training(ckpt)
print(f"Resumed run saved to: {res.save_dir}")
print("--- AUDIT & INSPECTION END ---")

csv = res.save_dir / "results.csv"
if csv.exists():
    df = pd.read_csv(csv); df.columns=[c.strip() for c in df.columns]
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    plt.figure(figsize=(8,4))
    for col in loss_cols: plt.plot(df["epoch"], df[col], label=col, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss After Resume")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

del base
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""},
"th": {
"md": """\
# 🧠 EX54: การฝึกสอนต่อจากจุดที่ถูกขัดจังหวะ (Resuming Interrupted Training)

การเทรนอาจถูกขัดจังหวะด้วย OOM, ไฟดับ หรือ cloud preemption
`last.pt` บันทึก **สถานะการเทรนทั้งหมด** ไว้ครบถ้วน

## สิ่งที่อยู่ภายใน `last.pt`
```python
{
  'epoch':        42,     # รอบล่าสุดที่เสร็จ (นับจาก 0)
  'best_fitness': 0.712,  # คะแนน fitness ดีที่สุด
  'model':        ...,    # weight state_dict
  'ema':          ...,    # EMA state_dict (→ best.pt)
  'optimizer':    ...,    # tensor m_t, v_t ของ AdamW
  'train_args':   {...},  # config เดิม
}
```

### ทำไม Optimizer State สำคัญ?
$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t \\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2$$
การสูญเสีย $m_t, v_t$ รีเซ็ต adaptive LR → ประสิทธิภาพตกชั่วคราว

- ✅ ใช้ `last.pt` เสมอ
- ❌ อย่าใช้ `best.pt` (ขาด optimizer state)

## 🔗 ลิงก์
- [[EX53_Training_Settings_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import torch, pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solution import resume_yolo_training
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")
print("\\n--- เริ่มการตรวจสอบ ---")

# ขั้นที่ 1: สร้าง checkpoint ด้วยการเทรน 2 epoch
print("[ขั้นที่ 1] เทรน 2 epoch เพื่อสร้าง last.pt...")
base = YOLO("yolo11n.pt")
base_res = base.train(data="coco8.yaml", epochs=2, imgsz=320, batch=8, device=device, verbose=False)
ckpt = str(base_res.save_dir / "weights" / "last.pt")
print(f"Checkpoint: {ckpt}")

# ขั้นที่ 2: ตรวจสอบ checkpoint
print("\\n[ขั้นที่ 2] ตรวจสอบ checkpoint...")
ck = torch.load(ckpt, map_location="cpu")
print(f"  Keys: {list(ck.keys())}")
print(f"  Epoch เสร็จ: {ck.get('epoch','N/A')}")
print(f"  Best fitness: {ck.get('best_fitness',0.0):.4f}")
print(f"  Optimizer state: {'มี ✅' if ck.get('optimizer') is not None else 'ขาดหาย ❌'}")
ta = ck.get("train_args", {})
print(f"  Config: epochs={ta.get('epochs','?')}, batch={ta.get('batch','?')}")

# ขั้นที่ 3: Resume
print("\\n[ขั้นที่ 3] กำลัง resume...")
res = resume_yolo_training(ckpt)
print(f"บันทึกผลที่: {res.save_dir}")
print("--- สิ้นสุดการตรวจสอบ ---")

csv = res.save_dir / "results.csv"
if csv.exists():
    df = pd.read_csv(csv); df.columns=[c.strip() for c in df.columns]
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    plt.figure(figsize=(8,4))
    for col in loss_cols: plt.plot(df["epoch"], df[col], label=col, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss หลัง Resume")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

del base
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""}}

NOTEBOOKS["EX55_Evaluation_Modes"] = {
"en": {
"md": """\
# 🧠 EX55: Evaluation Modes & Detection Metrics

## IoU — Intersection over Union
$$\\text{IoU}(\\hat{B}, B_{gt}) = \\frac{|\\hat{B}\\cap B_{gt}|}{|\\hat{B}\\cup B_{gt}|}$$
- IoU ≥ 0.5 → TP  |  IoU < 0.5 → FP

## Precision & Recall
$$P = \\frac{TP}{TP+FP} \\qquad R = \\frac{TP}{TP+FN}$$

## Average Precision (AP)
101-point COCO interpolation of the PR curve:
$$\\text{AP} = \\sum_k (R_k - R_{k-1})\\,P_k$$

## mAP@0.5 vs mAP@0.5:0.95
| Metric | Description |
|--------|-------------|
| mAP@0.5 | Mean AP across classes @ IoU=0.50 |
| mAP@0.5:0.95 | Mean AP averaged over IoU 0.50–0.95 (COCO standard) |

High mAP@0.5 + low mAP@0.5:0.95 = imprecise box localisation.

## 🔗 Links
- [[EX54_Resuming_Training_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, torch
import matplotlib.pyplot as plt
from solution import validate_model
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

print("\\n--- AUDIT & INSPECTION START ---")
metrics = validate_model("yolo11n.pt", "coco8.yaml")

print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall:    {metrics.box.mr:.4f}")
print(f"  mAP@0.5:   {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")

print("\\n  Per-class AP:")
names = metrics.names
for idx, name in names.items():
    p, r, ap50, ap = metrics.box.class_result(idx)
    print(f"    [{idx:2d}] {name:<20} P={p:.3f} R={r:.3f} AP50={ap50:.3f} AP={ap:.3f}")
print("--- AUDIT & INSPECTION END ---")

cls_names = list(names.values())
ap50_vals = [metrics.box.class_result(i)[2] for i in names.keys()]
plt.figure(figsize=(10,4))
plt.bar(cls_names, ap50_vals, color="#2ecc71", edgecolor="black", linewidth=0.5)
plt.axhline(metrics.box.map50, color="red", linestyle="--", linewidth=1.5,
            label=f"Mean mAP@0.5 = {metrics.box.map50:.3f}")
plt.xlabel("Class"); plt.ylabel("AP@0.5")
plt.title("Per-Class AP@0.5 (yolo11n / coco8)")
plt.xticks(rotation=45, ha="right"); plt.legend(); plt.tight_layout(); plt.show()

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""},
"th": {
"md": """\
# 🧠 EX55: โหมดการประเมินผล (Evaluation Modes & Detection Metrics)

## IoU — Intersection over Union
$$\\text{IoU}(\\hat{B}, B_{gt}) = \\frac{|\\hat{B}\\cap B_{gt}|}{|\\hat{B}\\cup B_{gt}|}$$
- IoU ≥ 0.5 → TP (ตรวจจับถูก)  |  IoU < 0.5 → FP (ตรวจจับผิด)

## Precision & Recall
$$P = \\frac{TP}{TP+FP} \\qquad R = \\frac{TP}{TP+FN}$$

## Average Precision (AP)
101-point COCO interpolation ของ PR curve:
$$\\text{AP} = \\sum_k (R_k - R_{k-1})\\,P_k$$

## mAP@0.5 vs mAP@0.5:0.95
| ตัวชี้วัด | ความหมาย |
|----------|---------|
| mAP@0.5 | ค่าเฉลี่ย AP ทุกคลาส @ IoU=0.50 |
| mAP@0.5:0.95 | ค่าเฉลี่ย AP ที่ IoU 0.50–0.95 (มาตรฐาน COCO) |

mAP@0.5 สูง + mAP@0.5:0.95 ต่ำ = กล่องขอบเขตไม่แม่นยำ

## 🔗 ลิงก์
- [[EX54_Resuming_Training_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, torch
import matplotlib.pyplot as plt
from solution import validate_model
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

print("\\n--- เริ่มการตรวจสอบ ---")
metrics = validate_model("yolo11n.pt", "coco8.yaml")

print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall:    {metrics.box.mr:.4f}")
print(f"  mAP@0.5:   {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")

print("\\n  ผลแต่ละคลาส:")
names = metrics.names
for idx, name in names.items():
    p, r, ap50, ap = metrics.box.class_result(idx)
    print(f"    [{idx:2d}] {name:<20} P={p:.3f} R={r:.3f} AP50={ap50:.3f} AP={ap:.3f}")
print("--- สิ้นสุดการตรวจสอบ ---")

cls_names = list(names.values())
ap50_vals = [metrics.box.class_result(i)[2] for i in names.keys()]
plt.figure(figsize=(10,4))
plt.bar(cls_names, ap50_vals, color="#2ecc71", edgecolor="black", linewidth=0.5)
plt.axhline(metrics.box.map50, color="red", linestyle="--", linewidth=1.5,
            label=f"Mean mAP@0.5 = {metrics.box.map50:.3f}")
plt.xlabel("คลาส"); plt.ylabel("AP@0.5")
plt.title("AP@0.5 แต่ละคลาส (yolo11n / coco8)")
plt.xticks(rotation=45, ha="right"); plt.legend(); plt.tight_layout(); plt.show()

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""}}

NOTEBOOKS["EX56_Prediction_Sources"] = {
"en": {
"md": """\
# 🧠 EX56: Prediction Sources

`model.predict(source=...)` auto-selects a data loader:

| Source | Example | Loader |
|--------|---------|--------|
| File path | `"img.jpg"` | PIL/OpenCV |
| Directory | `"imgs/"` | glob+sort |
| URL | `"https://..."` | urllib |
| NumPy (HWC, BGR) | `np.ndarray` | direct buffer |
| PIL Image | `Image.open(...)` | numpy convert |
| Webcam | `0` | cv2.VideoCapture |
| RTSP | `"rtsp://..."` | cv2.VideoCapture |
| Video | `"video.mp4"` | cv2.VideoCapture |

**Pre-processing:** decode → letterbox → normalize [0→1] → CHW tensor

## 🔗 Links
- [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import cv2, torch, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from solution import run_inference
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# Synthetic test image
img_np = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img_np, "YOLO", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 4)
cv2.imwrite("src_test.jpg", img_np)

print("\\n--- AUDIT & INSPECTION START ---")
model = YOLO("yolo11n.pt")

print("[Source 1] File path...")
r1 = run_inference("yolo11n.pt", "src_test.jpg", save_results=True)
print(f"  Detections: {len(r1[0].boxes)}")

print("[Source 2] NumPy array (BGR HWC)...")
r2 = model.predict(source=img_np, verbose=False)
print(f"  Detections: {len(r2[0].boxes)}  | input shape={img_np.shape} dtype={img_np.dtype}")

print("[Source 3] PIL Image (RGB)...")
pil_img = Image.open("src_test.jpg")
r3 = model.predict(source=pil_img, verbose=False)
print(f"  Detections: {len(r3[0].boxes)}")
print("--- AUDIT & INSPECTION END ---")

fig, axes = plt.subplots(1, 3, figsize=(14,4))
for ax, (title, res) in zip(axes, [("File path",r1[0]),("NumPy",r2[0]),("PIL",r3[0])]):
    ax.imshow(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
    ax.set_title(title); ax.axis("off")
plt.suptitle("Same model — different source types"); plt.tight_layout(); plt.show()

del model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if os.path.exists("src_test.jpg"): os.remove("src_test.jpg")
"""},
"th": {
"md": """\
# 🧠 EX56: แหล่งข้อมูลสำหรับการทำนาย (Prediction Sources)

`model.predict(source=...)` เลือก data loader โดยอัตโนมัติ:

| แหล่ง | ตัวอย่าง | Loader |
|------|---------|--------|
| File path | `"img.jpg"` | PIL/OpenCV |
| Directory | `"imgs/"` | glob+sort |
| NumPy (HWC, BGR) | `np.ndarray` | direct buffer |
| PIL Image | `Image.open(...)` | numpy convert |
| Webcam | `0` | cv2.VideoCapture |
| RTSP | `"rtsp://..."` | cv2.VideoCapture |
| วิดีโอ | `"video.mp4"` | cv2.VideoCapture |

**Pre-processing:** decode → letterbox → normalize [0→1] → tensor CHW

## 🔗 ลิงก์
- [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import cv2, torch, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from solution import run_inference
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

img_np = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img_np, "YOLO", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 4)
cv2.imwrite("src_test.jpg", img_np)

print("\\n--- เริ่มการตรวจสอบ ---")
model = YOLO("yolo11n.pt")

print("[แหล่งที่ 1] File path...")
r1 = run_inference("yolo11n.pt", "src_test.jpg", save_results=True)
print(f"  พบ: {len(r1[0].boxes)} วัตถุ")

print("[แหล่งที่ 2] NumPy array...")
r2 = model.predict(source=img_np, verbose=False)
print(f"  พบ: {len(r2[0].boxes)} วัตถุ | shape={img_np.shape} dtype={img_np.dtype}")

print("[แหล่งที่ 3] PIL Image...")
r3 = model.predict(source=Image.open("src_test.jpg"), verbose=False)
print(f"  พบ: {len(r3[0].boxes)} วัตถุ")
print("--- สิ้นสุดการตรวจสอบ ---")

fig, axes = plt.subplots(1, 3, figsize=(14,4))
for ax, (title, res) in zip(axes, [("File path",r1[0]),("NumPy",r2[0]),("PIL",r3[0])]):
    ax.imshow(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
    ax.set_title(title); ax.axis("off")
plt.suptitle("โมเดลเดียวกัน — แหล่งข้อมูลต่างกัน"); plt.tight_layout(); plt.show()

del model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if os.path.exists("src_test.jpg"): os.remove("src_test.jpg")
"""}}

NOTEBOOKS["EX57_Bounding_Box_Parsing"] = {
"en": {
"md": """\
# 🧠 EX57: Bounding Box Parsing

```
results[0].boxes
 ├── .xyxy   [N,4]  pixel coords [x1,y1,x2,y2]
 ├── .xywh   [N,4]  pixel [cx,cy,w,h]
 ├── .xyxyn  [N,4]  normalized [0,1]
 ├── .cls    [N,1]  class int
 └── .conf   [N,1]  confidence float
```

**GPU→CPU rule:** Always `.cpu()` before `.numpy()`:
```python
boxes.xyxy.cpu().numpy()  # ✅
boxes.xyxy.numpy()        # ❌ RuntimeError on CUDA tensors
```

## 🔗 Links
- [[EX56_Prediction_Sources_TH]] | [[EX58_Model_Export_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solution import parse_yolo_results
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 80), (200, 300), (200, 200, 0), -1)
cv2.rectangle(img, (350, 100), (580, 380), (0, 200, 200), -1)
cv2.imwrite("bb_test.jpg", img)

print("\\n--- AUDIT & INSPECTION START ---")
model = YOLO("yolo11n.pt")
results = model.predict(source="bb_test.jpg", verbose=False)
boxes = results[0].boxes

print(f"  Detections: {len(boxes)}")
print(f"  Device:     {boxes.xyxy.device}")
print(f"  .xyxy shape:{boxes.xyxy.shape}  (N×4)")
print(f"  .cls  shape:{boxes.cls.shape}")
print(f"  .conf shape:{boxes.conf.shape}")

if len(boxes):
    cls_arr  = boxes.cls.cpu().numpy().astype(int).flatten()
    conf_arr = boxes.conf.cpu().numpy().flatten()
    xyxy_arr = boxes.xyxy.cpu().numpy()
    norm_arr = boxes.xyxyn.cpu().numpy()
    print("\\n  Detections:")
    for i,(cls,conf,xyxy,norm) in enumerate(zip(cls_arr,conf_arr,xyxy_arr,norm_arr)):
        print(f"    #{i+1} {model.names[cls]:<18} conf={conf:.4f}")
        print(f"         xyxy ={[f'{v:.1f}' for v in xyxy]}")
        print(f"         norm ={[f'{v:.3f}' for v in norm]}")
else:
    print("  No detections (expected for blank synthetic image).")

parsed = parse_yolo_results("yolo11n.pt", "bb_test.jpg")
print(f"\\n  solution.parse_yolo_results() → {len(parsed)} detections")
print("--- AUDIT & INSPECTION END ---")

plt.figure(figsize=(8,5))
plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
plt.title("Bounding Box Parsing — Annotated"); plt.axis("off"); plt.show()

del model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if os.path.exists("bb_test.jpg"): os.remove("bb_test.jpg")
"""},
"th": {
"md": """\
# 🧠 EX57: การแจงส่วนกล่องขอบเขต (Bounding Box Parsing)

```
results[0].boxes
 ├── .xyxy   [N,4]  พิกัดพิกเซล [x1,y1,x2,y2]
 ├── .xywh   [N,4]  พิกัดพิกเซล [cx,cy,w,h]
 ├── .xyxyn  [N,4]  normalized [0,1]
 ├── .cls    [N,1]  int คลาส
 └── .conf   [N,1]  float ความมั่นใจ
```

**กฎ GPU→CPU:** เรียก `.cpu()` ก่อน `.numpy()` เสมอ:
```python
boxes.xyxy.cpu().numpy()  # ✅
boxes.xyxy.numpy()        # ❌ RuntimeError บน CUDA
```

## 🔗 ลิงก์
- [[EX56_Prediction_Sources_TH]] | [[EX58_Model_Export_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solution import parse_yolo_results
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 80), (200, 300), (200, 200, 0), -1)
cv2.rectangle(img, (350, 100), (580, 380), (0, 200, 200), -1)
cv2.imwrite("bb_test.jpg", img)

print("\\n--- เริ่มการตรวจสอบ ---")
model = YOLO("yolo11n.pt")
results = model.predict(source="bb_test.jpg", verbose=False)
boxes = results[0].boxes

print(f"  Detections: {len(boxes)}")
print(f"  Device: {boxes.xyxy.device}")
print(f"  .xyxy: {boxes.xyxy.shape} | .cls: {boxes.cls.shape} | .conf: {boxes.conf.shape}")

if len(boxes):
    cls_arr  = boxes.cls.cpu().numpy().astype(int).flatten()
    conf_arr = boxes.conf.cpu().numpy().flatten()
    xyxy_arr = boxes.xyxy.cpu().numpy()
    norm_arr = boxes.xyxyn.cpu().numpy()
    print("\\n  ผลลัพธ์:")
    for i,(cls,conf,xyxy,norm) in enumerate(zip(cls_arr,conf_arr,xyxy_arr,norm_arr)):
        print(f"    #{i+1} {model.names[cls]:<18} conf={conf:.4f}")
        print(f"         xyxy ={[f'{v:.1f}' for v in xyxy]}")
        print(f"         norm ={[f'{v:.3f}' for v in norm]}")
else:
    print("  ไม่พบวัตถุ")

parsed = parse_yolo_results("yolo11n.pt", "bb_test.jpg")
print(f"\\n  solution.parse_yolo_results() → {len(parsed)} detection")
print("--- สิ้นสุดการตรวจสอบ ---")

plt.figure(figsize=(8,5))
plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
plt.title("Bounding Box Parsing — ผลลัพธ์ที่ Annotated"); plt.axis("off"); plt.show()

del model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if os.path.exists("bb_test.jpg"): os.remove("bb_test.jpg")
"""}}

NOTEBOOKS["EX58_Model_Export"] = {
"en": {
"md": """\
# 🧠 EX58: Model Export & Optimization

| Format | Benefit | API |
|--------|---------|-----|
| **ONNX** | Cross-platform runtime | `format="onnx"` |
| **TensorRT** | Max GPU throughput | `format="engine"` |
| **TFLite** | Mobile/Edge TPU | `format="tflite"` |
| **OpenVINO** | Intel CPU/NPU | `format="openvino"` |

- **half=True** → FP16: 2× throughput on Tensor Cores, ~50% file size
- **int8=True** → INT8: 4× compression, needs calibration images

> [!WARNING]
> TensorRT `.engine` is device-specific — compile on the exact deployment GPU.

## 🔗 Links
- [[EX57_Bounding_Box_Parsing_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os, time
import numpy as np, torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solution import export_yolo_model
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

print("\\n--- AUDIT & INSPECTION START ---")
print("Exporting yolo11n.pt → ONNX...")
onnx_path = export_yolo_model("yolo11n.pt", "onnx")
pt_mb   = os.path.getsize("yolo11n.pt") / 1024**2
onnx_mb = os.path.getsize(onnx_path) / 1024**2
print(f"  .pt={pt_mb:.2f} MB | .onnx={onnx_mb:.2f} MB")

# Latency benchmark: 10 warmup + 20 measured
print("\\nBenchmarking (10 warmup + 20 runs)...")
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model_pt   = YOLO("yolo11n.pt")
model_onnx = YOLO(onnx_path)

for _ in range(10):
    model_pt.predict(source=dummy, verbose=False)
    model_onnx.predict(source=dummy, verbose=False)

pt_t, onnx_t = [], []
for _ in range(20):
    t=time.perf_counter(); model_pt.predict(source=dummy, verbose=False);   pt_t.append((time.perf_counter()-t)*1000)
    t=time.perf_counter(); model_onnx.predict(source=dummy, verbose=False); onnx_t.append((time.perf_counter()-t)*1000)

print(f"  PyTorch: {np.mean(pt_t):.1f} ± {np.std(pt_t):.1f} ms")
print(f"  ONNX:    {np.mean(onnx_t):.1f} ± {np.std(onnx_t):.1f} ms")
print("--- AUDIT & INSPECTION END ---")

fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].bar([".pt",".onnx"],[pt_mb,onnx_mb],color=["#e74c3c","#3498db"])
axes[0].set_ylabel("Size (MB)"); axes[0].set_title("File Size")
axes[1].boxplot([pt_t,onnx_t],labels=["PyTorch","ONNX"])
axes[1].set_ylabel("Latency (ms)"); axes[1].set_title("Inference Latency (20 runs)")
plt.tight_layout(); plt.show()

del model_pt, model_onnx
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""},
"th": {
"md": """\
# 🧠 EX58: การส่งออกโมเดล (Model Export & Optimization)

| รูปแบบ | ประโยชน์ | API |
|--------|---------|-----|
| **ONNX** | ทุก platform | `format="onnx"` |
| **TensorRT** | GPU throughput สูงสุด | `format="engine"` |
| **TFLite** | Mobile/Edge TPU | `format="tflite"` |
| **OpenVINO** | Intel CPU/NPU | `format="openvino"` |

- **half=True** → FP16: throughput 2× บน Tensor Core, ขนาดลด ~50%
- **int8=True** → INT8: บีบอัด 4×, ต้องใช้ภาพ calibration

> [!WARNING]
> TensorRT `.engine` ขึ้นตรงกับ hardware — compile บนเครื่องที่จะ deploy เสมอ

## 🔗 ลิงก์
- [[EX57_Bounding_Box_Parsing_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os, time
import numpy as np, torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solution import export_yolo_model
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

print("\\n--- เริ่มการตรวจสอบ ---")
print("Export yolo11n.pt → ONNX...")
onnx_path = export_yolo_model("yolo11n.pt", "onnx")
pt_mb   = os.path.getsize("yolo11n.pt") / 1024**2
onnx_mb = os.path.getsize(onnx_path) / 1024**2
print(f"  .pt={pt_mb:.2f} MB | .onnx={onnx_mb:.2f} MB")

print("\\nวัด latency (10 warmup + 20 runs)...")
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model_pt = YOLO("yolo11n.pt"); model_onnx = YOLO(onnx_path)
for _ in range(10):
    model_pt.predict(source=dummy, verbose=False)
    model_onnx.predict(source=dummy, verbose=False)

pt_t, onnx_t = [], []
for _ in range(20):
    t=time.perf_counter(); model_pt.predict(source=dummy, verbose=False);   pt_t.append((time.perf_counter()-t)*1000)
    t=time.perf_counter(); model_onnx.predict(source=dummy, verbose=False); onnx_t.append((time.perf_counter()-t)*1000)

print(f"  PyTorch: {np.mean(pt_t):.1f} ± {np.std(pt_t):.1f} ms")
print(f"  ONNX:    {np.mean(onnx_t):.1f} ± {np.std(onnx_t):.1f} ms")
print("--- สิ้นสุดการตรวจสอบ ---")

fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].bar([".pt",".onnx"],[pt_mb,onnx_mb],color=["#e74c3c","#3498db"])
axes[0].set_ylabel("ขนาด (MB)"); axes[0].set_title("ขนาดไฟล์")
axes[1].boxplot([pt_t,onnx_t],labels=["PyTorch","ONNX"])
axes[1].set_ylabel("Latency (ms)"); axes[1].set_title("Benchmark (20 runs)")
plt.tight_layout(); plt.show()

del model_pt, model_onnx
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""}}

NOTEBOOKS["EX59_Multi_Task_Modes"] = {
"en": {
"md": """\
# 🧠 EX59: Multi-Task Modes

YOLO11 shares Backbone+Neck across tasks; only prediction Heads differ.

| Task | Suffix | Output attr | Example use |
|------|--------|-------------|-------------|
| Detect | `.pt` | `.boxes` | Vehicle counting |
| Segment | `-seg.pt` | `.masks` | Medical area measurement |
| Classify | `-cls.pt` | `.probs` | Defect classification |
| Pose | `-pose.pt` | `.keypoints` | Sports biomechanics |
| OBB | `-obb.pt` | `.obb` | Aerial imagery |

**Cost order:** Classify < Detect < Pose < OBB ≈ Segment

## 🔗 Links
- [[EX58_Model_Export_TH]] | [[EX60_Streaming_Inference_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from solution import get_yolo_model_by_task
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

dummy = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(dummy, (100,100), (400,350), (200,200,200), -1)

tasks = {
    "detect":   lambda r: f"boxes={len(r.boxes)}",
    "segment":  lambda r: f"masks={'present' if r.masks else 'None'}",
    "classify": lambda r: f"top1={r.probs.top1} conf={r.probs.top1conf.item():.3f}",
    "pose":     lambda r: f"kp_shape={r.keypoints.xy.shape if r.keypoints else 'None'}",
}

print("\\n--- AUDIT & INSPECTION START ---")
results_map = {}
for task, summarize in tasks.items():
    print(f"\\n[{task.upper()}]")
    model = get_yolo_model_by_task(task)
    res = model.predict(source=dummy, verbose=False)
    results_map[task] = res[0]
    print(f"  Summary: {summarize(res[0])}")
    if task=="detect" and res[0].boxes:
        print(f"  boxes.xyxy: {res[0].boxes.xyxy.shape}")
    elif task=="segment" and res[0].masks:
        print(f"  masks.data: {res[0].masks.data.shape}")
    elif task=="classify" and res[0].probs:
        print(f"  probs.data: {res[0].probs.data.shape}")
    elif task=="pose" and res[0].keypoints:
        print(f"  kp.xy: {res[0].keypoints.xy.shape}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print("\\n--- AUDIT & INSPECTION END ---")

fig, axes = plt.subplots(1, len(results_map), figsize=(4*len(results_map),4))
for ax,(task,result) in zip(axes, results_map.items()):
    ax.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
    ax.set_title(task.capitalize()); ax.axis("off")
plt.suptitle("YOLO11 Multi-Task Results"); plt.tight_layout(); plt.show()
"""},
"th": {
"md": """\
# 🧠 EX59: โหมดการทำงานหลายภารกิจ (Multi-Task Modes)

YOLO11 แชร์ Backbone+Neck เดียวกัน เฉพาะ prediction Head ต่างกัน

| ภารกิจ | นามสกุล | แอตทริบิวต์ | ตัวอย่างการใช้ |
|--------|--------|------------|--------------|
| Detect | `.pt` | `.boxes` | นับยานพาหนะ |
| Segment | `-seg.pt` | `.masks` | วัดพื้นที่ทางการแพทย์ |
| Classify | `-cls.pt` | `.probs` | จำแนกของเสีย |
| Pose | `-pose.pt` | `.keypoints` | วิเคราะห์ท่าทางกีฬา |
| OBB | `-obb.pt` | `.obb` | ภาพดาวเทียม |

**ลำดับต้นทุน:** Classify < Detect < Pose < OBB ≈ Segment

## 🔗 ลิงก์
- [[EX58_Model_Export_TH]] | [[EX60_Streaming_Inference_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from solution import get_yolo_model_by_task
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

dummy = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(dummy, (100,100), (400,350), (200,200,200), -1)

tasks = {
    "detect":   lambda r: f"boxes={len(r.boxes)}",
    "segment":  lambda r: f"masks={'มี' if r.masks else 'None'}",
    "classify": lambda r: f"top1={r.probs.top1} conf={r.probs.top1conf.item():.3f}",
    "pose":     lambda r: f"kp={r.keypoints.xy.shape if r.keypoints else 'None'}",
}

print("\\n--- เริ่มการตรวจสอบ ---")
results_map = {}
for task, summarize in tasks.items():
    print(f"\\n[{task.upper()}]")
    model = get_yolo_model_by_task(task)
    res = model.predict(source=dummy, verbose=False)
    results_map[task] = res[0]
    print(f"  สรุป: {summarize(res[0])}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print("--- สิ้นสุดการตรวจสอบ ---")

fig, axes = plt.subplots(1, len(results_map), figsize=(4*len(results_map),4))
for ax,(task,result) in zip(axes, results_map.items()):
    ax.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
    ax.set_title(task.capitalize()); ax.axis("off")
plt.suptitle("YOLO11 Multi-Task Results"); plt.tight_layout(); plt.show()
"""}}

NOTEBOOKS["EX60_Streaming_Inference"] = {
"en": {
"md": """\
# 🧠 EX60: Streaming Inference

`stream=False` collects all frames into a list → O(N) RAM → OOM on long video.
`stream=True` returns a Python **generator** → O(1) RAM constant.

| | `stream=False` | `stream=True` |
|--|--|--|
| Return | `list[Results]` | `Generator[Results]` |
| RAM | O(N frames) | O(1) |
| Indexable | ✅ `[5]` | ❌ |
| Use-case | Short clips | CCTV 24/7 |

**Frame-drop:** Use `vid_stride=N` to process every N-th frame for real-time throughput.

## 🔗 Links
- [[EX59_Multi_Task_Modes_TH]] | [[EX61_YOLO_Dataset_Format_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os, types
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from solution import stream_inference_generator
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# 30-frame synthetic video
video_path = "stream_demo.mp4"
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320,240))
for i in range(30):
    frame = np.zeros((240,320,3), dtype=np.uint8)
    cv2.circle(frame, (min(40+i*8,300),120), 25, (255,255,255), -1)
    cv2.putText(frame, f"F{i:02d}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
    out.write(frame)
out.release()

print("\\n--- AUDIT & INSPECTION START ---")
gen = stream_inference_generator("yolo11n.pt", video_path)
print(f"Generator type: {type(gen).__name__} | is_generator={isinstance(gen, types.GeneratorType)}")

counts = []
for idx, dets in enumerate(gen):
    counts.append(len(dets))
    if idx < 5 or dets:
        print(f"  Frame {idx:02d}: {len(dets)} dets"
              + (f" | conf={dets[0]['confidence']:.3f}" if dets else ""))

print(f"\\nTotal frames: {len(counts)}  | Total dets: {sum(counts)}")
print("RAM: O(1) constant — generator yields 1 frame at a time ✅")
print("--- AUDIT & INSPECTION END ---")

plt.figure(figsize=(10,4))
plt.bar(range(len(counts)), counts, color="#9b59b6", alpha=0.8)
plt.xlabel("Frame"); plt.ylabel("Detections")
plt.title("Detection Density over Video Stream")
plt.tight_layout(); plt.show()

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if os.path.exists(video_path): os.remove(video_path)
"""},
"th": {
"md": """\
# 🧠 EX60: การทำนายผลแบบสตรีมมิ่ง (Streaming Inference)

`stream=False` เก็บทุกเฟรมในลิสต์ → RAM แบบ O(N) → OOM กับวิดีโอยาว
`stream=True` คืน Python **generator** → RAM แบบ O(1) คงที่

| | `stream=False` | `stream=True` |
|--|--|--|
| ประเภทคืนค่า | `list[Results]` | `Generator[Results]` |
| RAM | O(N frames) | O(1) |
| เข้าถึงด้วย index | ✅ `[5]` | ❌ |
| เหมาะกับ | คลิปสั้น | กล้อง CCTV 24/7 |

**Frame-drop:** ใช้ `vid_stride=N` ประมวลผลทุก N เฟรม เพื่อ real-time throughput

## 🔗 ลิงก์
- [[EX59_Multi_Task_Modes_TH]] | [[EX61_YOLO_Dataset_Format_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os, types
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from solution import stream_inference_generator
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

video_path = "stream_demo.mp4"
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320,240))
for i in range(30):
    frame = np.zeros((240,320,3), dtype=np.uint8)
    cv2.circle(frame, (min(40+i*8,300),120), 25, (255,255,255), -1)
    cv2.putText(frame, f"F{i:02d}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
    out.write(frame)
out.release()

print("\\n--- เริ่มการตรวจสอบ ---")
gen = stream_inference_generator("yolo11n.pt", video_path)
print(f"ประเภท: {type(gen).__name__} | is_generator={isinstance(gen, types.GeneratorType)}")

counts = []
for idx, dets in enumerate(gen):
    counts.append(len(dets))
    if idx < 5 or dets:
        print(f"  เฟรม {idx:02d}: {len(dets)} วัตถุ"
              + (f" | conf={dets[0]['confidence']:.3f}" if dets else ""))

print(f"\\nรวม {len(counts)} เฟรม | detections: {sum(counts)}")
print("RAM: O(1) — generator ประมวลผลทีละเฟรม ✅")
print("--- สิ้นสุดการตรวจสอบ ---")

plt.figure(figsize=(10,4))
plt.bar(range(len(counts)), counts, color="#9b59b6", alpha=0.8)
plt.xlabel("เฟรม"); plt.ylabel("จำนวน Detections")
plt.title("ความหนาแน่นการตรวจจับในวิดีโอ")
plt.tight_layout(); plt.show()

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if os.path.exists(video_path): os.remove(video_path)
"""}}

NOTEBOOKS["EX61_YOLO_Dataset_Format"] = {
"en": {
"md": """\
# 🧠 EX61: YOLO Dataset Format & Normalized Coordinate Conversion

Each image has one `.txt` label file with **normalized** coordinates.

## Math
Given pixel bbox $[x_{min}, y_{min}, x_{max}, y_{max}]$ on image $W\\times H$:
$$x_c = \\frac{x_{min}+x_{max}}{2W},\\quad y_c = \\frac{y_{min}+y_{max}}{2H},\\quad w = \\frac{x_{max}-x_{min}}{W},\\quad h = \\frac{y_{max}-y_{min}}{H}$$

Label line: `<class_id> <x_c> <y_c> <w> <h>` (all values in [0,1])

## data.yaml
```yaml
path: /abs/path/dataset
train: train/images
val:   val/images
nc: 3
names: {0: cat, 1: dog, 2: bird}
```

## 🔗 Links
- [[EX62_Inference_Visualization_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from solution import convert_to_yolo_format
%matplotlib inline

W, H = 640, 480
test_cases = [
    ([120, 160, 360, 400], 0, "cat"),
    ([400, 50, 600, 250], 1, "dog"),
    ([10, 10, 100, 80],   2, "bird"),
]

print("\\n--- AUDIT & INSPECTION START ---")
label_lines = []
for bbox, class_id, name in test_cases:
    yolo_str = convert_to_yolo_format(bbox, W, H, class_id)
    p = yolo_str.split()
    cid, xc, yc, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])

    assert 0<=xc<=1 and 0<=yc<=1 and 0<=bw<=1 and 0<=bh<=1, f"Out of [0,1] for {name}!"
    print(f"  {name} | pixel={bbox}")
    print(f"    YOLO: {yolo_str}")
    print(f"    Audit: xc={xc:.4f} yc={yc:.4f} w={bw:.4f} h={bh:.4f} ✅")
    label_lines.append(yolo_str)

# Write real .txt annotation file
label_path = Path("audit_label.txt")
label_path.write_text("\\n".join(label_lines) + "\\n")
print(f"\\n✅ Wrote {label_path}: {label_path.stat().st_size} bytes")
print(f"Contents:\\n{label_path.read_text()}")

# Round-trip decode
print("Round-trip decode:")
for line in label_path.read_text().strip().split("\\n"):
    p = line.split()
    cid2, xc2, yc2, bw2, bh2 = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
    x1=(xc2-bw2/2)*W; y1=(yc2-bh2/2)*H; x2=(xc2+bw2/2)*W; y2=(yc2+bh2/2)*H
    print(f"  class={cid2} → pixel xyxy=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
print("--- AUDIT & INSPECTION END ---")

# Visualise
fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(np.ones((H,W,3), dtype=np.uint8)*230)
colors = ["#e74c3c","#3498db","#2ecc71"]
for (bbox,cid,name),color in zip(test_cases, colors):
    x1,y1,x2,y2 = bbox
    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1, linewidth=2, edgecolor=color, facecolor=color, alpha=0.25))
    ax.text(x1,y1-5, name, color=color, fontsize=10, fontweight="bold")
ax.set_xlim(0,W); ax.set_ylim(H,0)
ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
ax.set_title("YOLO Coordinate Conversion Visualization")
plt.tight_layout(); plt.show()

label_path.unlink(missing_ok=True)
"""},
"th": {
"md": """\
# 🧠 EX61: รูปแบบชุดข้อมูล YOLO (Dataset Format & Coordinate Conversion)

แต่ละภาพมีไฟล์ `.txt` label ที่ใช้พิกัด **normalized**

## คณิตศาสตร์
กำหนด pixel bbox $[x_{min}, y_{min}, x_{max}, y_{max}]$ บนภาพขนาด $W\\times H$:
$$x_c = \\frac{x_{min}+x_{max}}{2W},\\quad y_c = \\frac{y_{min}+y_{max}}{2H},\\quad w = \\frac{x_{max}-x_{min}}{W},\\quad h = \\frac{y_{max}-y_{min}}{H}$$

บรรทัด label: `<class_id> <x_c> <y_c> <w> <h>` (ค่าทั้งหมดอยู่ใน [0,1])

## data.yaml
```yaml
path: /abs/path/dataset
train: train/images
val:   val/images
nc: 3
names: {0: cat, 1: dog, 2: bird}
```

## 🔗 ลิงก์
- [[EX62_Inference_Visualization_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from solution import convert_to_yolo_format
%matplotlib inline

W, H = 640, 480
test_cases = [
    ([120, 160, 360, 400], 0, "cat"),
    ([400, 50, 600, 250], 1, "dog"),
    ([10, 10, 100, 80],   2, "bird"),
]

print("\\n--- เริ่มการตรวจสอบ ---")
label_lines = []
for bbox, class_id, name in test_cases:
    yolo_str = convert_to_yolo_format(bbox, W, H, class_id)
    p = yolo_str.split()
    cid, xc, yc, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
    assert 0<=xc<=1 and 0<=yc<=1 and 0<=bw<=1 and 0<=bh<=1, f"พิกัดเกินขอบเขต {name}!"
    print(f"  {name} | pixel={bbox}")
    print(f"    YOLO: {yolo_str}")
    print(f"    ตรวจสอบ: xc={xc:.4f} yc={yc:.4f} w={bw:.4f} h={bh:.4f} ✅")
    label_lines.append(yolo_str)

label_path = Path("audit_label.txt")
label_path.write_text("\\n".join(label_lines) + "\\n")
print(f"\\n✅ เขียน {label_path}: {label_path.stat().st_size} bytes")
print(f"เนื้อหา:\\n{label_path.read_text()}")

print("แปลงกลับ (round-trip):")
for line in label_path.read_text().strip().split("\\n"):
    p = line.split()
    cid2, xc2, yc2, bw2, bh2 = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
    x1=(xc2-bw2/2)*W; y1=(yc2-bh2/2)*H; x2=(xc2+bw2/2)*W; y2=(yc2+bh2/2)*H
    print(f"  class={cid2} → pixel xyxy=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
print("--- สิ้นสุดการตรวจสอบ ---")

fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(np.ones((H,W,3), dtype=np.uint8)*230)
colors = ["#e74c3c","#3498db","#2ecc71"]
for (bbox,cid,name),color in zip(test_cases, colors):
    x1,y1,x2,y2 = bbox
    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1, linewidth=2, edgecolor=color, facecolor=color, alpha=0.25))
    ax.text(x1,y1-5, name, color=color, fontsize=10, fontweight="bold")
ax.set_xlim(0,W); ax.set_ylim(H,0)
ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
ax.set_title("การแปลงพิกัด YOLO Dataset Format")
plt.tight_layout(); plt.show()

label_path.unlink(missing_ok=True)
"""}}

NOTEBOOKS["EX62_Inference_Visualization"] = {
"en": {
"md": """\
# 🧠 EX62: Inference Visualization

`results[0].plot()` renders predictions → BGR NumPy array (HWC uint8).

| param | default | effect |
|-------|---------|--------|
| `conf` | True | show confidence |
| `labels` | True | show class name |
| `boxes` | True | draw bbox |
| `line_width` | auto | border thickness |

**RGB vs BGR:** OpenCV = BGR; `plot()` returns BGR.
Always `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before Matplotlib.

## 🔗 Links
- [[EX57_Bounding_Box_Parsing_TH]] | [[EX63_Multi_Object_Tracking_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from solution import visualize_and_save
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (50,80), (250,300), (80,180,255), -1)
cv2.rectangle(img, (350,120), (590,390), (255,100,50), -1)
cv2.imwrite("vis_in.jpg", img)

print("\\n--- AUDIT & INSPECTION START ---")
visualize_and_save("yolo11n.pt", "vis_in.jpg", "vis_out.jpg")

if os.path.exists("vis_out.jpg"):
    out = cv2.imread("vis_out.jpg")
    print(f"  Output shape: {out.shape} | dtype: {out.dtype}")
    print(f"  Unique colours: {len(np.unique(out.reshape(-1,3), axis=0))}")
    print("  ✅ Output file verified")
else:
    print("  ❌ Output not found")
print("--- AUDIT & INSPECTION END ---")

fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Input"); axes[0].axis("off")
if os.path.exists("vis_out.jpg"):
    axes[1].imshow(cv2.cvtColor(cv2.imread("vis_out.jpg"), cv2.COLOR_BGR2RGB))
    axes[1].set_title("YOLO plot() output"); axes[1].axis("off")
plt.suptitle("Before vs After YOLO Visualization")
plt.tight_layout(); plt.show()

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
for f in ["vis_in.jpg","vis_out.jpg"]:
    if os.path.exists(f): os.remove(f)
"""},
"th": {
"md": """\
# 🧠 EX62: การแสดงผลการทำนาย (Inference Visualization)

`results[0].plot()` วาด prediction → NumPy array แบบ BGR (HWC uint8)

| พารามิเตอร์ | ค่าเริ่มต้น | ผลกระทบ |
|-----------|-----------|---------|
| `conf` | True | แสดง confidence |
| `labels` | True | แสดงชื่อคลาส |
| `boxes` | True | วาดกล่องขอบเขต |
| `line_width` | auto | ความหนาเส้น |

**RGB vs BGR:** OpenCV ใช้ BGR; `plot()` คืน BGR
`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` ก่อนใช้กับ Matplotlib เสมอ

## 🔗 ลิงก์
- [[EX57_Bounding_Box_Parsing_TH]] | [[EX63_Multi_Object_Tracking_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from solution import visualize_and_save
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (50,80), (250,300), (80,180,255), -1)
cv2.rectangle(img, (350,120), (590,390), (255,100,50), -1)
cv2.imwrite("vis_in.jpg", img)

print("\\n--- เริ่มการตรวจสอบ ---")
visualize_and_save("yolo11n.pt", "vis_in.jpg", "vis_out.jpg")

if os.path.exists("vis_out.jpg"):
    out = cv2.imread("vis_out.jpg")
    print(f"  ขนาด output: {out.shape} | dtype: {out.dtype}")
    print("  ✅ ตรวจสอบไฟล์เรียบร้อย")
else:
    print("  ❌ ไม่พบไฟล์ output")
print("--- สิ้นสุดการตรวจสอบ ---")

fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("ต้นฉบับ"); axes[0].axis("off")
if os.path.exists("vis_out.jpg"):
    axes[1].imshow(cv2.cvtColor(cv2.imread("vis_out.jpg"), cv2.COLOR_BGR2RGB))
    axes[1].set_title("YOLO plot() output"); axes[1].axis("off")
plt.suptitle("ก่อน vs หลัง YOLO Visualization")
plt.tight_layout(); plt.show()

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
for f in ["vis_in.jpg","vis_out.jpg"]:
    if os.path.exists(f): os.remove(f)
"""}}

NOTEBOOKS["EX63_Multi_Object_Tracking"] = {
"en": {
"md": """\
# 🧠 EX63: Multi-Object Tracking (MOT)

Tracking assigns persistent IDs across frames via:
1. YOLO detects boxes at frame $t$
2. **Kalman Filter** predicts each track's position at $t+1$
3. **Hungarian algorithm** on IoU matrix links predictions → detections
4. Unmatched detections → new IDs; unmatched tracks → deleted

| Feature | BoT-SORT (default) | ByteTrack |
|---------|--------------------|-----------|
| Re-ID | ✅ appearance | ❌ motion only |
| Camera motion | ✅ CMC | ❌ |
| Speed | Moderate | Fast |
| Best for | Moving cameras | Dense crowd |

**Key:** Set `persist=True` in frame-by-frame loops — without it, tracker state resets every call.

## 🔗 Links
- [[EX60_Streaming_Inference_TH]] | [[EX62_Inference_Visualization_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from solution import track_objects_in_video
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# 2-object synthetic tracking video
video_path = "tracking_demo.mp4"
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640,480))
traj = {0:[], 1:[]}
for i in range(20):
    frame = np.zeros((480,640,3), dtype=np.uint8)
    cx1 = 100 + i*18; cx2 = 540 - i*18
    cv2.circle(frame, (cx1,200), 40, (255,180,50), -1)
    cv2.circle(frame, (cx2,280), 40, (50,180,255), -1)
    cv2.putText(frame, f"F{i:02d}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    traj[0].append(cx1); traj[1].append(cx2)
    out.write(frame)
out.release()

print("\\n--- AUDIT & INSPECTION START ---")
track_ids = track_objects_in_video("yolo11n.pt", video_path, tracker_config="bytetrack.yaml")
print(f"  Unique track IDs: {track_ids}")
print(f"  Count: {len(track_ids)} distinct objects")
print("--- AUDIT & INSPECTION END ---")

fig, axes = plt.subplots(1,2,figsize=(12,4))
frames = list(range(20))
axes[0].plot(frames, traj[0], "r-o", label="Object A"); axes[0].plot(frames, traj[1], "b-o", label="Object B")
axes[0].set_xlabel("Frame"); axes[0].set_ylabel("X position"); axes[0].set_title("GT Trajectories")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].bar(range(len(track_ids)), [1]*len(track_ids), tick_label=[f"ID {t}" for t in track_ids], color="#9b59b6")
axes[1].set_title(f"Track IDs ({len(track_ids)} unique)"); axes[1].set_ylabel("Active")
plt.tight_layout(); plt.show()

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if os.path.exists(video_path): os.remove(video_path)
"""},
"th": {
"md": """\
# 🧠 EX63: การติดตามวัตถุหลายชิ้น (Multi-Object Tracking)

การติดตามกำหนด persistent ID ข้ามเฟรม ผ่าน:
1. YOLO ตรวจจับ box ที่เฟรม $t$
2. **Kalman Filter** ทำนายตำแหน่ง track ที่เฟรม $t+1$
3. **Hungarian algorithm** บน IoU matrix จับคู่ prediction → detection
4. Detection ที่ไม่มีคู่ → ID ใหม่; Track ที่หาย → ถูกลบ

| คุณสมบัติ | BoT-SORT (default) | ByteTrack |
|----------|---------------------|-----------|
| Re-ID | ✅ ลักษณะภาพ | ❌ การเคลื่อนที่เท่านั้น |
| ชดเชยกล้อง | ✅ CMC | ❌ |
| ความเร็ว | ปานกลาง | เร็ว |
| เหมาะกับ | กล้องเคลื่อนที่ | ฝูงชนหนาแน่น |

**สำคัญ:** ต้องตั้ง `persist=True` ในลูปเฟรมต่อเฟรม — หากไม่ตั้ง tracker จะรีเซ็ตทุกครั้ง

## 🔗 ลิงก์
- [[EX60_Streaming_Inference_TH]] | [[EX62_Inference_Visualization_TH]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, os
import cv2, numpy as np, torch
import matplotlib.pyplot as plt
from solution import track_objects_in_video
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

video_path = "tracking_demo.mp4"
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640,480))
traj = {0:[], 1:[]}
for i in range(20):
    frame = np.zeros((480,640,3), dtype=np.uint8)
    cx1 = 100 + i*18; cx2 = 540 - i*18
    cv2.circle(frame, (cx1,200), 40, (255,180,50), -1)
    cv2.circle(frame, (cx2,280), 40, (50,180,255), -1)
    cv2.putText(frame, f"F{i:02d}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    traj[0].append(cx1); traj[1].append(cx2)
    out.write(frame)
out.release()

print("\\n--- เริ่มการตรวจสอบ ---")
track_ids = track_objects_in_video("yolo11n.pt", video_path, tracker_config="bytetrack.yaml")
print(f"  Track IDs: {track_ids}")
print(f"  จำนวน: {len(track_ids)} วัตถุ")
print("--- สิ้นสุดการตรวจสอบ ---")

fig, axes = plt.subplots(1,2,figsize=(12,4))
frames = list(range(20))
axes[0].plot(frames, traj[0], "r-o", label="วัตถุ A"); axes[0].plot(frames, traj[1], "b-o", label="วัตถุ B")
axes[0].set_xlabel("เฟรม"); axes[0].set_ylabel("ตำแหน่ง X"); axes[0].set_title("แนววิถีการเคลื่อนที่")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].bar(range(len(track_ids)), [1]*len(track_ids), tick_label=[f"ID {t}" for t in track_ids], color="#9b59b6")
axes[1].set_title(f"Track IDs ({len(track_ids)} unique)"); axes[1].set_ylabel("Active")
plt.tight_layout(); plt.show()

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
if os.path.exists(video_path): os.remove(video_path)
"""}}

NOTEBOOKS["EX64_Callbacks_and_Logging"] = {
"en": {
"md": """\
# 🧠 EX64: Callback Hooks & Custom Metric Logging

YOLO fires Python callables at lifecycle events — you add logic without modifying YOLO internals.

| Hook | Fires when |
|------|-----------|
| `on_train_start` | Before first epoch |
| `on_train_epoch_end` | After train loss, **before** val |
| `on_fit_epoch_end` | After **both** train + val |
| `on_val_end` | After standalone `model.val()` |
| `on_train_end` | After all epochs |

**Key:** Use `on_fit_epoch_end` (not `on_train_epoch_end`) to read `metrics/mAP50(B)` —
that key is only available **after** validation runs.

## 🔗 Links
- [[EX53_Training_Settings_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solution import register_custom_metric_callback
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

print("\\n--- AUDIT & INSPECTION START ---")
model = YOLO("yolo11n.pt")
print(f"Pre-reg 'on_val_end': {model.callbacks.get('on_val_end', [])}")

model = register_custom_metric_callback(model)
post_cbs = model.callbacks.get("on_val_end", [])
print(f"Post-reg 'on_val_end': {[cb.__name__ for cb in post_cbs]}")
print(f"Total hooks across all events: {sum(len(v) for v in model.callbacks.values())}")

print("\\nRunning model.val() on coco8 to trigger callback...")
try:
    results = model.val(data="coco8.yaml", imgsz=320, device=device, verbose=False)
    mAP50 = results.box.map50
    mAP95 = results.box.map
    mp = results.box.mp
    mr = results.box.mr
    print(f"  Precision={mp:.4f}  Recall={mr:.4f}")
    print(f"  mAP@0.5={mAP50:.4f}  mAP@0.5:0.95={mAP95:.4f}")
except Exception as e:
    print(f"Validation error: {e}")
    mAP50=mAP95=mp=mr=0.0
print("--- AUDIT & INSPECTION END ---")

fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].bar(["mAP@0.5","mAP@0.5:0.95"], [mAP50,mAP95], color=["#27ae60","#145a32"], edgecolor="black")
axes[0].set_ylim(0,1.0); axes[0].set_ylabel("Score"); axes[0].set_title("mAP (via callback)")
for i,v in enumerate([mAP50,mAP95]): axes[0].text(i, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
axes[1].bar(["Precision","Recall"], [mp,mr], color=["#2980b9","#8e44ad"], edgecolor="black")
axes[1].set_ylim(0,1.0); axes[1].set_ylabel("Score"); axes[1].set_title("P & R")
for i,v in enumerate([mp,mr]): axes[1].text(i, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout(); plt.show()

del model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""},
"th": {
"md": """\
# 🧠 EX64: Callback Hooks และการบันทึก Metric แบบกำหนดเอง

YOLO เรียกใช้ callable Python ณ จุดสำคัญในวงจรการเทรน

| Hook | ทำงานเมื่อ |
|------|-----------|
| `on_train_start` | ก่อน epoch แรก |
| `on_train_epoch_end` | หลัง train loss, **ก่อน** val |
| `on_fit_epoch_end` | หลัง **ทั้ง** train + val |
| `on_val_end` | หลัง `model.val()` |
| `on_train_end` | หลังทุก epoch |

**สำคัญ:** ใช้ `on_fit_epoch_end` (ไม่ใช่ `on_train_epoch_end`) เพื่ออ่าน `metrics/mAP50(B)` —
key นั้นจะมีค่า **หลัง** validation เท่านั้น

## 🔗 ลิงก์
- [[EX53_Training_Settings_TH]] | [[YOLO_Learning_Plan]]
""",
"code": """\
# Back up or checkpoint this section of code before starting to modify the large file.
import gc, torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from solution import register_custom_metric_callback
%matplotlib inline

device = "0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] ใช้อุปกรณ์: {device}")

print("\\n--- เริ่มการตรวจสอบ ---")
model = YOLO("yolo11n.pt")
print(f"ก่อนลงทะเบียน: {model.callbacks.get('on_val_end', [])}")

model = register_custom_metric_callback(model)
post_cbs = model.callbacks.get("on_val_end", [])
print(f"หลังลงทะเบียน: {[cb.__name__ for cb in post_cbs]}")
print(f"Hooks ทั้งหมด: {sum(len(v) for v in model.callbacks.values())}")

print("\\nรัน model.val() บน coco8 เพื่อ trigger callback...")
try:
    results = model.val(data="coco8.yaml", imgsz=320, device=device, verbose=False)
    mAP50=results.box.map50; mAP95=results.box.map; mp=results.box.mp; mr=results.box.mr
    print(f"  Precision={mp:.4f}  Recall={mr:.4f}")
    print(f"  mAP@0.5={mAP50:.4f}  mAP@0.5:0.95={mAP95:.4f}")
except Exception as e:
    print(f"Validation error: {e}")
    mAP50=mAP95=mp=mr=0.0
print("--- สิ้นสุดการตรวจสอบ ---")

fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].bar(["mAP@0.5","mAP@0.5:0.95"],[mAP50,mAP95],color=["#27ae60","#145a32"],edgecolor="black")
axes[0].set_ylim(0,1.0); axes[0].set_ylabel("คะแนน"); axes[0].set_title("mAP (ผ่าน callback)")
for i,v in enumerate([mAP50,mAP95]): axes[0].text(i, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
axes[1].bar(["Precision","Recall"],[mp,mr],color=["#2980b9","#8e44ad"],edgecolor="black")
axes[1].set_ylim(0,1.0); axes[1].set_ylabel("คะแนน"); axes[1].set_title("P & R")
for i,v in enumerate([mp,mr]): axes[1].text(i, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout(); plt.show()

del model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""}}

# ── Generation loop ────────────────────────────────────────────────────────────
generated, skipped = [], []
for ex_dir, langs in NOTEBOOKS.items():
    ex_path = BASE / ex_dir
    if not ex_path.is_dir():
        print(f"⚠  {ex_dir} not found, skipping")
        skipped.append(ex_dir)
        continue
    for lang, content in langs.items():
        suffix = "" if lang=="en" else "_TH"
        nb_file = ex_path / f"explanation{suffix}.ipynb"
        nb_file.write_text(json.dumps(nb(content["md"], content["code"]), indent=1, ensure_ascii=False))
        generated.append(str(nb_file.relative_to(BASE)))
        print(f"  ✅  {nb_file.relative_to(BASE)}")

print(f"\n{'='*60}")
print(f"Generated: {len(generated)}")
if skipped: print(f"Skipped:   {skipped}")
print("Done.")
