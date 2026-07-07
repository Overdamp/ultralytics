import json
import shutil
from pathlib import Path

# Print the required safety/backup warning
print("Back up or checkpoint this section of code before starting to modify the large file.")

base_dir = Path("/home/luke/ai_training/ultralytics/learning_lab/experiments/04_basic_yolo")

# Notebook data definitions
notebooks_content = {
    "EX51_YOLO_CLI": {
        "en": {
            "title": "🧠 EX51: YOLO Command Line Interface (CLI)",
            "markdown": """# 🧠 EX51: YOLO Command Line Interface (CLI)

The Command Line Interface (CLI) is one of the most powerful and convenient ways to interact with Ultralytics YOLO. It allows you to train, validate, predict, export, track, and benchmark models directly from the terminal.

## CLI Syntax Structure
The standard command syntax is:
```bash
yolo TASK MODE ARGS
```

### Components:
1. **`yolo`**: The entrypoint command.
2. **`TASK`**: The vision task (e.g., `detect`, `segment`, `classify`, `pose`, `obb`). Optional if inferred from model suffix (e.g., `-seg`).
3. **`MODE`**: The phase (e.g., `train`, `val`, `predict`, `export`, `track`, `benchmark`).
4. **`ARGS`**: Custom arguments in the form of `key=value` pairs.

## ⚠️ Safety Warning
Before running any training commands via the CLI, always verify the paths and configurations in your dataset definition file (`data.yaml`). You can use a file viewer like `view_file` or check it manually to ensure absolute path mappings are correct and avoid configuration errors during training.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
import os
import glob
import cv2
import matplotlib.pyplot as plt
from solution import run_yolo_cli_command
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
print("Executing YOLO CLI via subprocess...")
result = run_yolo_cli_command(
    task="detect",
    mode="predict",
    model="yolov8n.pt",
    data=None,
    epochs=None,
    imgsz=320
)

print(f"Subprocess return code: {result.returncode}")
print(f"Command output sample:\\n{result.stdout[-500:]}")
print("--- AUDIT & INSPECTION END ---")

# Plot prediction results
predict_dirs = sorted(glob.glob("runs/detect/predict*"))
if predict_dirs:
    latest_dir = predict_dirs[-1]
    saved_images = glob.glob(os.path.join(latest_dir, "*.jpg"))
    if saved_images:
        img_path = saved_images[0]
        print(f"Loading and plotting prediction: {img_path}")
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tensor Auditing: Print image shape and metadata
        print(f"Image tensor shape: {img.shape}")
        print(f"Pixel values range: min={img.min()}, max={img.max()}")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(f"YOLO CLI Prediction: {os.path.basename(img_path)}")
        plt.axis("off")
        plt.show()
    else:
        print("No prediction images found in the output directory.")
else:
    print("No predict directories found.")"""
        },
        "th": {
            "title": "🧠 EX51: YOLO Command Line Interface (CLI) (อินเตอร์เฟซบรรทัดคำสั่ง)",
            "markdown": """# 🧠 EX51: YOLO Command Line Interface (CLI) (อินเตอร์เฟซบรรทัดคำสั่ง)

Command Line Interface (CLI) เป็นหนึ่งในวิธีที่มีประสิทธิภาพและสะดวกที่สุดในการใช้งาน Ultralytics YOLO ซึ่งช่วยให้คุณสามารถฝึกฝน (train), ตรวจสอบ (validate), ทำนาย (predict), ส่งออก (export), ติดตามวัตถุ (track) และวัดประสิทธิภาพ (benchmark) โมเดลได้โดยตรงจากเทอร์มินัลโดยไม่ต้องเขียนโค้ด Python เลยแม้แต่บรรทัดเดียว

## โครงสร้างไวยากรณ์ CLI (CLI Syntax Structure)
ไวยากรณ์มาตรฐานของคำสั่งคือ:
```bash
yolo TASK MODE ARGS
```

### ส่วนประกอบสำคัญ:
1. **`yolo`**: คำสั่งหลักเริ่มต้น (Entrypoint command)
2. **`TASK`**: งานด้านคอมพิวเตอร์วิทัศน์ (เช่น `detect`, `segment`, `classify`, `pose`, `obb`) ซึ่งระบุหรือไม่ระบุก็ได้ หากระบุรุ่นของโมเดลที่มีคำต่อท้ายชัดเจน (เช่น `-seg` จะระบุโดยอัตโนมัติว่าเป็นงาน Instance Segmentation)
3. **`MODE`**: ขั้นตอนการทำงานที่ต้องการดำเนินการ (เช่น `train`, `val`, `predict`, `export`, `track`, `benchmark`)
4. **`ARGS`**: พารามิเตอร์เสริมต่าง ๆ ในรูปแบบคู่ `key=value`

## ⚠️ คำเตือนความปลอดภัย
ก่อนที่จะรันคำสั่งสำหรับฝึกฝนโมเดลผ่าน CLI ให้ตรวจสอบไฟล์ตั้งค่าชุดข้อมูล (`data.yaml`) เสมอเพื่อความถูกต้องของโครงสร้างเส้นทางไดเรกทอรี โดยสามารถใช้โปรแกรมดูไฟล์อย่าง `view_file` เพื่อความปลอดภัยในการเตรียมข้อมูล

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
import os
import glob
import cv2
import matplotlib.pyplot as plt
from solution import run_yolo_cli_command
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
print("Executing YOLO CLI via subprocess...")
result = run_yolo_cli_command(
    task="detect",
    mode="predict",
    model="yolov8n.pt",
    data=None,
    epochs=None,
    imgsz=320
)

print(f"Subprocess return code: {result.returncode}")
print(f"Command output sample:\\n{result.stdout[-500:]}")
print("--- AUDIT & INSPECTION END ---")

# Plot prediction results
predict_dirs = sorted(glob.glob("runs/detect/predict*"))
if predict_dirs:
    latest_dir = predict_dirs[-1]
    saved_images = glob.glob(os.path.join(latest_dir, "*.jpg"))
    if saved_images:
        img_path = saved_images[0]
        print(f"Loading and plotting prediction: {img_path}")
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tensor Auditing: Print image shape and metadata
        print(f"Image tensor shape: {img.shape}")
        print(f"Pixel values range: min={img.min()}, max={img.max()}")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(f"YOLO CLI Prediction: {os.path.basename(img_path)}")
        plt.axis("off")
        plt.show()
    else:
        print("No prediction images found in the output directory.")
else:
    print("No predict directories found.")"""
        }
    },
    "EX52_Model_Configurations": {
        "en": {
            "title": "🧠 EX52: Model Configurations",
            "markdown": """# 🧠 EX52: Model Configurations

This notebook demonstrates the difference between building a YOLO model from scratch using a configuration file (`yolov8n.yaml`) versus loading a pretrained weight model (`yolov8n.pt`).

## 1. Scratch (YAML) vs. Pretrained (.pt)
- **Scratch (YAML)**: The model is initialized with random weights according to the architecture defined in the YAML file. This is useful for training from scratch on custom datasets when the target distribution is highly distinct from standard datasets.
- **Pretrained (.pt)**: The model is loaded with weights that have already been optimized on a large dataset (e.g. COCO). This is ideal for transfer learning to reduce training time and improve generalization.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
import torch
import matplotlib.pyplot as plt
from solution import load_model_from_yaml, load_pretrained_model
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
# Load model from scratch configuration
yaml_model = load_model_from_yaml("yolov8n.yaml")
# Load model with pre-trained weights
pretrained_model = load_pretrained_model("yolov8n.pt")

# Auditing Model Architectures
print(f"YAML Model architecture type: {type(yaml_model.model)}")
print(f"Pretrained Model architecture type: {type(pretrained_model.model)}")

yaml_params = sum(p.numel() for p in yaml_model.model.parameters())
pretrained_params = sum(p.numel() for p in pretrained_model.model.parameters())
print(f"YAML Model total parameters: {yaml_params:,}")
print(f"Pretrained Model total parameters: {pretrained_params:,}")

# Audit tensor shapes using a dummy input
dummy_input = torch.randn(1, 3, 320, 320)
print(f"Dummy input shape: {dummy_input.shape}")

yaml_model.model.eval()
pretrained_model.model.eval()
with torch.no_grad():
    yaml_out = yaml_model.model(dummy_input)
    pretrained_out = pretrained_model.model(dummy_input)

# In YOLOv8, output is list, tuple or tensor. Let's print shape details
print(f"YAML model output type: {type(yaml_out)}")
if isinstance(yaml_out, (list, tuple)):
    print(f"Number of output heads/tensors: {len(yaml_out)}")
    for idx, t in enumerate(yaml_out):
        if isinstance(t, torch.Tensor):
            print(f"  Head {idx} tensor shape: {t.shape}")
        elif isinstance(t, (list, tuple)):
            print(f"  Head {idx} is list/tuple of length {len(t)}")
            for idx2, t2 in enumerate(t):
                if isinstance(t2, torch.Tensor):
                    print(f"    Sub-tensor {idx2} shape: {t2.shape}")
else:
    print(f"Model output shape: {yaml_out.shape}")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Plot parameter comparison
plt.figure(figsize=(6, 4))
plt.bar(["Random (YAML)", "Pretrained (PT)"], [yaml_params, pretrained_params], color=['red', 'blue'])
plt.ylabel("Number of Parameters")
plt.title("Model Parameters: Random Config vs Pretrained Weights")
plt.show()"""
        },
        "th": {
            "title": "🧠 EX52: Model Configurations (การกำหนดค่าคอนฟิกโมเดล)",
            "markdown": """# 🧠 EX52: Model Configurations (การกำหนดค่าคอนฟิกโมเดล)

สมุดบันทึกนี้แสดงความแตกต่างระหว่างการโหลดโมเดลจากโครงสร้างเริ่มต้น (YAML configuration - สุ่มค่าน้ำหนักเริ่มต้น) กับการโหลดโมเดลที่มีการฝึกฝนมาก่อนแล้ว (Pretrained weights - .pt)

## 1. Scratch (YAML) vs. Pretrained (.pt)
- **Scratch (YAML)**: โมเดลจะสร้างขึ้นโดยมีน้ำหนักสุ่มตามรูปแบบสถาปัตยกรรมโครงข่ายประสาทเทียมที่ประกาศไว้ในไฟล์ YAML เหมาะสำหรับในกรณีที่ต้องการฝึกฝนโมเดลใหม่ทั้งหมด
- **Pretrained (.pt)**: โหลดน้ำหนักที่มีการฝึกฝนบนชุดข้อมูลขนาดใหญ่มาก (เช่น COCO dataset) มาเรียบร้อยแล้ว เหมาะสำหรับทำ Transfer learning เพื่อลดเวลาและสร้างการทำนายที่มีความแม่นยำสูงอย่างรวดเร็ว

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
import torch
import matplotlib.pyplot as plt
from solution import load_model_from_yaml, load_pretrained_model
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
# Load model from scratch configuration
yaml_model = load_model_from_yaml("yolov8n.yaml")
# Load model with pre-trained weights
pretrained_model = load_pretrained_model("yolov8n.pt")

# Auditing Model Architectures
print(f"YAML Model architecture type: {type(yaml_model.model)}")
print(f"Pretrained Model architecture type: {type(pretrained_model.model)}")

yaml_params = sum(p.numel() for p in yaml_model.model.parameters())
pretrained_params = sum(p.numel() for p in pretrained_model.model.parameters())
print(f"YAML Model total parameters: {yaml_params:,}")
print(f"Pretrained Model total parameters: {pretrained_params:,}")

# Audit tensor shapes using a dummy input
dummy_input = torch.randn(1, 3, 320, 320)
print(f"Dummy input shape: {dummy_input.shape}")

yaml_model.model.eval()
pretrained_model.model.eval()
with torch.no_grad():
    yaml_out = yaml_model.model(dummy_input)
    pretrained_out = pretrained_model.model(dummy_input)

# In YOLOv8, output is list, tuple or tensor. Let's print shape details
print(f"YAML model output type: {type(yaml_out)}")
if isinstance(yaml_out, (list, tuple)):
    print(f"Number of output heads/tensors: {len(yaml_out)}")
    for idx, t in enumerate(yaml_out):
        if isinstance(t, torch.Tensor):
            print(f"  Head {idx} tensor shape: {t.shape}")
        elif isinstance(t, (list, tuple)):
            print(f"  Head {idx} is list/tuple of length {len(t)}")
            for idx2, t2 in enumerate(t):
                if isinstance(t2, torch.Tensor):
                    print(f"    Sub-tensor {idx2} shape: {t2.shape}")
else:
    print(f"Model output shape: {yaml_out.shape}")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Plot parameter comparison
plt.figure(figsize=(6, 4))
plt.bar(["Random (YAML)", "Pretrained (PT)"], [yaml_params, pretrained_params], color=['red', 'blue'])
plt.ylabel("Number of Parameters")
plt.title("Model Parameters: Random Config vs Pretrained Weights")
plt.show()"""
        }
    },
    "EX53_Training_Settings": {
        "en": {
            "title": "🧠 EX53: Training Settings",
            "markdown": """# 🧠 EX53: Training Settings

This exercise focuses on setting hyperparameters (batch size, image size, learning rate, optimization) and configuring YOLO models for training on a custom dataset.

## ⚠️ Safety Check before Training
**Always verify the dataset path configurations in `data.yaml` before starting a training command.** Incorrect path mappings will cause training to fail or load incorrect images. Use a command like `view_file` or check it manually to make sure directories are correctly configured.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import train_custom_model
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
%matplotlib inline

print("[WARNING] Before running training, make sure the dataset YAML (e.g. coco8.yaml) is verified!")
print("--- AUDIT & INSPECTION START ---")
# We run 1 epoch on coco8 dataset for training demonstration
results = train_custom_model(
    model_path="yolov8n.pt",
    data_yaml_path="coco8.yaml",
    epochs=1,
    batch_size=8,
    imgsz=224,
    device="cpu"
)
print("Training Completed.")
print(f"Training Results save directory: {results.save_dir}")
print("--- AUDIT & INSPECTION END ---")

# Find the training results.csv and plot the loss
results_csv = os.path.join(results.save_dir, "results.csv")
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    print("Auditing logged losses in results.csv:")
    print(df[['train/box_loss', 'train/cls_loss', 'train/dfl_loss']].tail(1))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='orange')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', color='blue')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='orange', linestyle='--')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='blue', linestyle='--')
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    batch_imgs = glob.glob(os.path.join(results.save_dir, "train_batch*.jpg"))
    if batch_imgs:
        import cv2
        img = cv2.imread(batch_imgs[0])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title("Training Batch Labels")
        plt.axis("off")
        plt.show()
    else:
        print("Results files not found. Creating simulated loss plot.")
        epochs = [1]
        box_loss = [1.25]
        cls_loss = [0.85]
        plt.plot(epochs, box_loss, label='Box Loss')
        plt.plot(epochs, cls_loss, label='Cls Loss')
        plt.title('Simulated Losses (Epoch 1)')
        plt.legend()
        plt.show()"""
        },
        "th": {
            "title": "🧠 EX53: Training Settings (การกำหนดค่าการฝึกฝนโมเดล)",
            "markdown": """# 🧠 EX53: Training Settings (การกำหนดค่าการฝึกฝนโมเดล)

สมุดบันทึกนี้มีวัตถุประสงค์เพื่อศึกษาและปรับปรุงพารามิเตอร์ต่าง ๆ ในการฝึกฝน YOLO (เช่น ขนาดแบทช์, ขนาดความละเอียดรูปภาพ, อัตราการเรียนรู้ และวิธีการออพติไมซ์)

## ⚠️ การตรวจสอบความปลอดภัยก่อนเริ่มต้นฝึกฝนโมเดล
**ตรวจสอบความถูกต้องในไฟล์ตั้งค่าชุดข้อมูล `data.yaml` เสมอก่อนรันการเทรน** หากพาธรูปภาพไม่ถูกต้อง การฝึกฝนจะหยุดชะงักหรือไม่สามารถค้นหาข้อมูลได้ ให้ใช้เครื่องมือดูไฟล์อย่าง `view_file` ตรวจสอบให้มั่นใจก่อนลงมือเสมอ

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import train_custom_model
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
%matplotlib inline

print("[WARNING] Before running training, make sure the dataset YAML (e.g. coco8.yaml) is verified!")
print("--- AUDIT & INSPECTION START ---")
# We run 1 epoch on coco8 dataset for training demonstration
results = train_custom_model(
    model_path="yolov8n.pt",
    data_yaml_path="coco8.yaml",
    epochs=1,
    batch_size=8,
    imgsz=224,
    device="cpu"
)
print("Training Completed.")
print(f"Training Results save directory: {results.save_dir}")
print("--- AUDIT & INSPECTION END ---")

# Find the training results.csv and plot the loss
results_csv = os.path.join(results.save_dir, "results.csv")
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    print("Auditing logged losses in results.csv:")
    print(df[['train/box_loss', 'train/cls_loss', 'train/dfl_loss']].tail(1))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='orange')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', color='blue')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='orange', linestyle='--')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='blue', linestyle='--')
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    batch_imgs = glob.glob(os.path.join(results.save_dir, "train_batch*.jpg"))
    if batch_imgs:
        import cv2
        img = cv2.imread(batch_imgs[0])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title("Training Batch Labels")
        plt.axis("off")
        plt.show()
    else:
        print("Results files not found. Creating simulated loss plot.")
        epochs = [1]
        box_loss = [1.25]
        cls_loss = [0.85]
        plt.plot(epochs, box_loss, label='Box Loss')
        plt.plot(epochs, cls_loss, label='Cls Loss')
        plt.title('Simulated Losses (Epoch 1)')
        plt.legend()
        plt.show()"""
        }
    },
    "EX54_Resuming_Training": {
        "en": {
            "title": "🧠 EX54: Resuming Training",
            "markdown": """# 🧠 EX54: Resuming Training

This notebook explains how to resume interrupted training from a checkpoint (`last.pt`) without losing training progress.

## 1. Resume Training Logic
- **`resume=True`**: Tells the trainer to load the exact state of the optimizer, learning rate scheduler, and epoch counts from the checkpoint.
- **Safety checks**: The paths and config files must be identical to the original run. Verify dataset paths in `data.yaml` before continuing.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import resume_yolo_training
import os
import matplotlib.pyplot as plt
%matplotlib inline

print("[WARNING] Verify your checkpoint file (usually last.pt) and data.yaml exist before resuming training!")
checkpoint_file = "runs/detect/train/weights/last.pt"

print("--- AUDIT & INSPECTION START ---")
print(f"Resume checkpoint configuration check: checkpoint={checkpoint_file}")

if os.path.exists(checkpoint_file):
    print("Checkpoint found! Starting resume...")
    try:
        results = resume_yolo_training(checkpoint_file)
        print("Resumed training finished successfully.")
        print(f"Results saved in: {results.save_dir}")
    except Exception as e:
        print(f"Error resuming training: {e}")
else:
    print(f"Checkpoint file {checkpoint_file} not found. Skipping execution to prevent failure.")
    print("Simulation: Resumed training would load optimizer state, epoch index, and weights from state_dict.")
print("--- AUDIT & INSPECTION END ---")

# Visualizing Resume Progress (Simulated or Real)
plt.figure(figsize=(6, 4))
plt.bar(["Completed Epochs", "Remaining Epochs"], [5, 5], color=['green', 'blue'])
plt.ylabel("Epochs")
plt.title("Resuming Training: Training Epoch Progress")
plt.show()"""
        },
        "th": {
            "title": "🧠 EX54: Resuming Training (การฝึกฝนต่อจากจุดบันทึก)",
            "markdown": """# 🧠 EX54: Resuming Training (การฝึกฝนต่อจากจุดบันทึก)

สมุดบันทึกนี้สาธิตวิธีการทำงานของการฝึกฝนต่อ (resume) จากไฟล์จุดบันทึกผลลัพธ์การฝึกฝน (ปกติบันทึกในชื่อ `last.pt`) เพื่อประหยัดเวลาการเทรนที่ขาดตอน

## 1. ตรรกะเบื้องหลังการรัน Resume
- **`resume=True`**: ส่งสัญญาณให้ YOLO ค้นหาและเตรียมสถานะสุดท้ายของออพติไมเซอร์ (Optimizer), ตารางปรับลดอัตราการเรียนรู้ (LR Scheduler) และเลขเอพ็อคจากไฟล์ตรวจสอบ
- **ข้อพิจารณาความปลอดภัย**: คุณลักษณะของชุดข้อมูลและโครงสร้างพาธจะต้องเหมือนกับตอนเริ่มต้นอย่างละเอียด

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import resume_yolo_training
import os
import matplotlib.pyplot as plt
%matplotlib inline

print("[WARNING] Verify your checkpoint file (usually last.pt) and data.yaml exist before resuming training!")
checkpoint_file = "runs/detect/train/weights/last.pt"

print("--- AUDIT & INSPECTION START ---")
print(f"Resume checkpoint configuration check: checkpoint={checkpoint_file}")

if os.path.exists(checkpoint_file):
    print("Checkpoint found! Starting resume...")
    try:
        results = resume_yolo_training(checkpoint_file)
        print("Resumed training finished successfully.")
        print(f"Results saved in: {results.save_dir}")
    except Exception as e:
        print(f"Error resuming training: {e}")
else:
    print(f"Checkpoint file {checkpoint_file} not found. Skipping execution to prevent failure.")
    print("Simulation: Resumed training would load optimizer state, epoch index, and weights from state_dict.")
print("--- AUDIT & INSPECTION END ---")

# Visualizing Resume Progress (Simulated or Real)
plt.figure(figsize=(6, 4))
plt.bar(["Completed Epochs", "Remaining Epochs"], [5, 5], color=['green', 'blue'])
plt.ylabel("Epochs")
plt.title("Resuming Training: Training Epoch Progress")
plt.show()"""
        }
    },
    "EX55_Evaluation_Modes": {
        "en": {
            "title": "🧠 EX55: Evaluation Modes",
            "markdown": """# 🧠 EX55: Evaluation Modes

Validation is a critical step in assessing the performance of a trained model. It computes metrics like Precision, Recall, and mAP (Mean Average Precision) on a validation dataset split.

## Mathematical Equations and Metrics
1. **Precision**: Measures the ratio of true detections among all positive predictions.
   $$\\text{Precision} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}$$
2. **Recall**: Measures the ratio of true detections among all ground truth targets.
   $$\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}$$
3. **mAP@0.5**: The Mean Average Precision evaluated at an Intersection over Union (IoU) threshold of 0.5.
4. **mAP@0.5:0.95**: The Average mAP across IoU thresholds starting from 0.5 to 0.95 with steps of 0.05.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import validate_model
import matplotlib.pyplot as plt
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
print("Validating model on coco8.yaml...")
metrics = validate_model("yolov8n.pt", "coco8.yaml")

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("--- AUDIT & INSPECTION END ---")

# Plot metrics
plt.figure(figsize=(8, 4))
keys = list(metrics.keys())
values = list(metrics.values())
colors = ['blue', 'green', 'orange', 'red']

plt.bar(keys, values, color=colors)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("YOLO Model Validation Metrics on coco8.yaml")
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
plt.show()"""
        },
        "th": {
            "title": "🧠 EX55: Evaluation Modes (โหมดการวัดผลและการประเมินประสิทธิภาพ)",
            "markdown": """# 🧠 EX55: Evaluation Modes (โหมดการวัดผลและการประเมินประสิทธิภาพ)

การประเมินประสิทธิภาพเป็นขั้นตอนสำคัญเพื่อศึกษาและตรวจสอบคุณภาพของโมเดลผ่านการคำนวณมาตรวัดความแม่นยำ ได้แก่ Precision, Recall และ mAP บนชุดข้อมูลสำหรับทดสอบ (Validation set)

## นิยามและสมการทางคณิตศาสตร์
1. **Precision (ความแม่นยำของการพยากรณ์เชิงบวก)**:
   $$\\text{Precision} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}$$
2. **Recall (ความครอบคลุมข้อมูลจริง)**:
   $$\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}$$
3. **mAP@0.5**: ค่าเฉลี่ยความแม่นยำเฉลี่ย (Mean Average Precision) ที่คำนวณที่ระดับ Intersection over Union (IoU) เท่ากับ 0.5
4. **mAP@0.5:0.95**: ค่าเฉลี่ย mAP ที่เกณฑ์ IoU ต่าง ๆ ตั้งแต่ 0.5 ถึง 0.95 (ห่างช่วงละ 0.05)

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import validate_model
import matplotlib.pyplot as plt
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
print("Validating model on coco8.yaml...")
metrics = validate_model("yolov8n.pt", "coco8.yaml")

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("--- AUDIT & INSPECTION END ---")

# Plot metrics
plt.figure(figsize=(8, 4))
keys = list(metrics.keys())
values = list(metrics.values())
colors = ['blue', 'green', 'orange', 'red']

plt.bar(keys, values, color=colors)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("YOLO Model Validation Metrics on coco8.yaml")
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
plt.show()"""
        }
    },
    "EX56_Prediction_Sources": {
        "en": {
            "title": "🧠 EX56: Prediction Sources",
            "markdown": """# 🧠 EX56: Prediction Sources

YOLO models support inference on various input sources (images, video folders, directory of images, URLs, webcam indices, PyTorch tensors, NumPy arrays, PIL images).

## 1. Parameters in `.predict()`
- **`source`**: The input file/stream/array.
- **`save=True`**: Saves the annotated prediction image automatically to disk.
- **`conf`**: Confidence threshold for predictions (default 0.25).

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import run_inference
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Create a custom numpy source image
source_img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(source_img, (50, 50), (150, 150), (255, 0, 0), -1)   # Blue square
cv2.circle(source_img, (280, 150), 50, (0, 0, 255), -1)          # Red circle
cv2.imwrite("synthetic_source.jpg", source_img)

print("--- AUDIT & INSPECTION START ---")
print("Running prediction on synthetic image...")
results = run_inference("yolov8n.pt", "synthetic_source.jpg", save_results=True)

# Audit output results
first_result = results[0]
print(f"Results length: {len(results)}")
print(f"Image original shape: {first_result.orig_shape}")
print(f"Detected bounding boxes count: {len(first_result.boxes)}")
if len(first_result.boxes) > 0:
    print(f"Confidence scores: {first_result.boxes.conf.cpu().tolist()}")
    print(f"Class IDs: {first_result.boxes.cls.cpu().tolist()}")
    print(f"Bounding box shape: {first_result.boxes.xyxy.shape}")
print("--- AUDIT & INSPECTION END ---")

# Visualize original vs plotted detection results
annotated_img = first_result.plot()
annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(source_img_rgb)
plt.title("Original Synthetic Input")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(annotated_img_rgb)
plt.title("YOLO Detection Result")
plt.axis("off")
plt.show()

# Cleanup
import os
if os.path.exists("synthetic_source.jpg"):
    os.remove("synthetic_source.jpg")"""
        },
        "th": {
            "title": "🧠 EX56: Prediction Sources (แหล่งรับข้อมูลสำหรับการทำนายผล)",
            "markdown": """# 🧠 EX56: Prediction Sources (แหล่งรับข้อมูลสำหรับการทำนายผล)

โมเดล YOLO สามารถประมวลผลคำสั่งตรวจจับและทำนายผลบนข้อมูลอินพุตได้หลากหลายรูปแบบ (อาทิเช่น ไฟล์รูปภาพ, เส้นทางโฟลเดอร์ภาพ, ลิงก์ URL, สตรีมกล้องเว็บแคม, อาเรย์ NumPy หรือรูปภาพ PIL)

## 1. การเรียกใช้ `.predict()` และพารามิเตอร์เด่น
- **`source`**: ข้อมูลภาพนำเข้า
- **`save=True`**: บันทึกรูปภาพที่วาดกล่องล้อมรอบวัตถุลงบนดิสก์
- **`conf`**: ค่าเกณฑ์ความเชื่อมั่นในการตัดสินใจตรวจจับวัตถุ

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import run_inference
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Create a custom numpy source image
source_img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(source_img, (50, 50), (150, 150), (255, 0, 0), -1)   # Blue square
cv2.circle(source_img, (280, 150), 50, (0, 0, 255), -1)          # Red circle
cv2.imwrite("synthetic_source.jpg", source_img)

print("--- AUDIT & INSPECTION START ---")
print("Running prediction on synthetic image...")
results = run_inference("yolov8n.pt", "synthetic_source.jpg", save_results=True)

# Audit output results
first_result = results[0]
print(f"Results length: {len(results)}")
print(f"Image original shape: {first_result.orig_shape}")
print(f"Detected bounding boxes count: {len(first_result.boxes)}")
if len(first_result.boxes) > 0:
    print(f"Confidence scores: {first_result.boxes.conf.cpu().tolist()}")
    print(f"Class IDs: {first_result.boxes.cls.cpu().tolist()}")
    print(f"Bounding box shape: {first_result.boxes.xyxy.shape}")
print("--- AUDIT & INSPECTION END ---")

# Visualize original vs plotted detection results
annotated_img = first_result.plot()
annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(source_img_rgb)
plt.title("Original Synthetic Input")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(annotated_img_rgb)
plt.title("YOLO Detection Result")
plt.axis("off")
plt.show()

# Cleanup
import os
if os.path.exists("synthetic_source.jpg"):
    os.remove("synthetic_source.jpg")"""
        }
    },
    "EX57_Bounding_Box_Parsing": {
        "en": {
            "title": "🧠 EX57: Bounding Box Parsing",
            "markdown": """# 🧠 EX57: Bounding Box Parsing

Extracting and converting coordinate data from bounding boxes is standard practice in custom computer vision pipelines.

## Mathematical Mapping of Coordinates
- **`xyxy` (Corner Format)**:
  `[xmin, ymin, xmax, ymax]` represent the absolute pixel coordinates of the top-left and bottom-right corners.
- **`xywh` (Center-Size Format)**:
  `[x_center, y_center, width, height]` represent the center pixel and dimensions of the box.

$$\\text{width} = x_{\\max} - x_{\\min}$$
$$\\text{height} = y_{\\max} - y_{\\min}$$
$$x_{\\text{center}} = x_{\\min} + \\frac{\\text{width}}{2}$$
$$y_{\\text{center}} = y_{\\min} + \\frac{\\text{height}}{2}$$

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import parse_yolo_results
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Create a test image with a drawing
img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.circle(img, (300, 200), 70, (0, 255, 0), -1) # Draw a green circle
cv2.imwrite("box_test.jpg", img)

print("--- AUDIT & INSPECTION START ---")
# Predict using pretrained model
model = YOLO("yolov8n.pt")
results = model.predict("box_test.jpg", verbose=False)

# Parse prediction coordinates
parsed = parse_yolo_results(results)
print(f"Total parsed detections: {len(parsed)}")

for i, det in enumerate(parsed):
    print(f"Detection {i}:")
    print(f"  Class ID: {det['class_id']}")
    print(f"  Confidence: {det['confidence']:.4f}")
    print(f"  xyxy corners: {det['bbox_xyxy']}")
    print(f"  xywh center/size: {det['bbox_xywh']}")
    
    # Audit coordinate consistency: xywh from xyxy
    xmin, ymin, xmax, ymax = det['bbox_xyxy']
    xc, yc, w, h = det['bbox_xywh']
    calc_xc = (xmin + xmax) / 2.0
    calc_yc = (ymin + ymax) / 2.0
    calc_w = xmax - xmin
    calc_h = ymax - ymin
    print(f"  Verification check:")
    print(f"    xc diff: {abs(calc_xc - xc):.6f}, yc diff: {abs(calc_yc - yc):.6f}")
    print(f"    width diff: {abs(calc_w - w):.6f}, height diff: {abs(calc_h - h):.6f}")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Draw using matplotlib custom bounding box labels
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax = plt.gca()

for det in parsed:
    xmin, ymin, xmax, ymax = det['bbox_xyxy']
    rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.text(xmin, ymin - 10, f"Class {det['class_id']} ({det['confidence']:.2f})", 
            bbox=dict(facecolor='red', alpha=0.5), color='white', fontsize=12)

plt.title("Parsed Custom Bounding Box Visualization")
plt.axis("off")
plt.show()

# Cleanup
import os
if os.path.exists("box_test.jpg"):
    os.remove("box_test.jpg")"""
        },
        "th": {
            "title": "🧠 EX57: Bounding Box Parsing (การถอดวิเคราะห์รหัสพิกัดกล่องล้อมรอบวัตถุ)",
            "markdown": """# 🧠 EX57: Bounding Box Parsing (การถอดวิเคราะห์รหัสพิกัดกล่องล้อมรอบวัตถุ)

การแกะโครงสร้างพิกัดและความน่าจะเป็นในการตรวจจับมีความสำคัญในกระบวนการทำงานแบบกำหนดเอง (Custom Pipelines)

## สูตรการคำนวณสลับพิกัดกรอบวัตถุ
- **`xyxy` (พิกัดตามมุมขอบ)**:
  `[xmin, ymin, xmax, ymax]` แสดงตำแหน่งพิกเซลจากจุดมุมซ้ายบนไปยังมุมขวาล่าง
- **`xywh` (พิกัดตามจุดกึ่งกลางและขนาด)**:
  `[x_center, y_center, width, height]` แสดงพิกัดกึ่งกลาง ขนาดความกว้างและความสูง

$$\\text{width} = x_{\\max} - x_{\\min}$$
$$\\text{height} = y_{\\max} - y_{\\min}$$
$$x_{\\text{center}} = x_{\\min} + \\frac{\\text{width}}{2}$$
$$y_{\\text{center}} = y_{\\min} + \\frac{\\text{height}}{2}$$

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import parse_yolo_results
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Create a test image with a drawing
img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.circle(img, (300, 200), 70, (0, 255, 0), -1) # Draw a green circle
cv2.imwrite("box_test.jpg", img)

print("--- AUDIT & INSPECTION START ---")
# Predict using pretrained model
model = YOLO("yolov8n.pt")
results = model.predict("box_test.jpg", verbose=False)

# Parse prediction coordinates
parsed = parse_yolo_results(results)
print(f"Total parsed detections: {len(parsed)}")

for i, det in enumerate(parsed):
    print(f"Detection {i}:")
    print(f"  Class ID: {det['class_id']}")
    print(f"  Confidence: {det['confidence']:.4f}")
    print(f"  xyxy corners: {det['bbox_xyxy']}")
    print(f"  xywh center/size: {det['bbox_xywh']}")
    
    # Audit coordinate consistency: xywh from xyxy
    xmin, ymin, xmax, ymax = det['bbox_xyxy']
    xc, yc, w, h = det['bbox_xywh']
    calc_xc = (xmin + xmax) / 2.0
    calc_yc = (ymin + ymax) / 2.0
    calc_w = xmax - xmin
    calc_h = ymax - ymin
    print(f"  Verification check:")
    print(f"    xc diff: {abs(calc_xc - xc):.6f}, yc diff: {abs(calc_yc - yc):.6f}")
    print(f"    width diff: {abs(calc_w - w):.6f}, height diff: {abs(calc_h - h):.6f}")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Draw using matplotlib custom bounding box labels
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax = plt.gca()

for det in parsed:
    xmin, ymin, xmax, ymax = det['bbox_xyxy']
    rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.text(xmin, ymin - 10, f"Class {det['class_id']} ({det['confidence']:.2f})", 
            bbox=dict(facecolor='red', alpha=0.5), color='white', fontsize=12)

plt.title("Parsed Custom Bounding Box Visualization")
plt.axis("off")
plt.show()

# Cleanup
import os
if os.path.exists("box_test.jpg"):
    os.remove("box_test.jpg")"""
        }
    },
    "EX58_Model_Export": {
        "en": {
            "title": "🧠 EX58: Model Export",
            "markdown": """# 🧠 EX58: Model Export

Exporting a model formats weights and computational graphs into formats suited for fast inference engines (like ONNX, TensorRT/Engine, CoreML, TFLite).

## 1. Benefits of ONNX format
- **Interoperability**: ONNX models can run on different frameworks and devices (CPU, GPU, Edge devices).
- **Latency**: Graph optimization reduces computational redundancy and speeds up predictions.
- **Half Precision (FP16)**: Reduces model size by 50% and speeds up inference on compatible GPU architectures.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import export_yolo_model
import torch
import time
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
# Export model to ONNX format
onnx_path = export_yolo_model("yolov8n.pt", "onnx")
print(f"Exported model path: {onnx_path}")

pt_size = os.path.getsize("yolov8n.pt") / (1024 * 1024)
onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"PyTorch model size: {pt_size:.2f} MB")
print(f"ONNX model size: {onnx_size:.2f} MB")

# Benchmarking latency
dummy_input = torch.randn(1, 3, 640, 640)
model_pt = YOLO("yolov8n.pt")

t0 = time.time()
for _ in range(10):
    model_pt.predict(dummy_input, verbose=False)
pt_latency = (time.time() - t0) * 1000 / 10.0

model_onnx = YOLO(onnx_path)
t0 = time.time()
for _ in range(10):
    model_onnx.predict(dummy_input, verbose=False)
onnx_latency = (time.time() - t0) * 1000 / 10.0

print(f"PyTorch Average Inference Latency: {pt_latency:.2f} ms")
print(f"ONNX Average Inference Latency: {onnx_latency:.2f} ms")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Latency comparison
plt.figure(figsize=(6, 4))
plt.bar(["PyTorch", "ONNX"], [pt_latency, onnx_latency], color=['blue', 'green'])
plt.ylabel("Latency (ms)")
plt.title("Inference Latency: PyTorch vs. ONNX")
plt.show()"""
        },
        "th": {
            "title": "🧠 EX58: Model Export (การส่งออกโมเดล)",
            "markdown": """# 🧠 EX58: Model Export (การส่งออกโมเดล)

การส่งออกโมเดลคือขั้นตอนการแปลงน้ำหนักโมเดลและกราฟการคำนวณให้เป็นรูปแบบมาตรฐานอื่น ๆ สำหรับนำไปเปิดใช้งานเพื่อทำนายแบบเร็ว (เช่น ONNX, TensorRT, CoreML, TFLite)

## 1. จุดเด่นของรูปแบบ ONNX
- **สถาปัตยกรรมทำงานร่วมกัน**: โค้ด ONNX สามารถทำงานบนอุปกรณ์ CPU, GPU และอุปกรณ์ Edge ได้ข้ามเฟรมเวิร์ก
- **ลดดีเลย์ (Latency)**: โครงสร้างกราฟที่สมบูรณ์จะกำจัดการคำนวณที่เกินความจำเป็น
- **ความแม่นยำครึ่งเดียว (FP16)**: ช่วยลดขนาดลงเกือบครึ่งและประหยัดการประมวลผลบนการ์ดจอที่รองรับ

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import export_yolo_model
import torch
import time
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
# Export model to ONNX format
onnx_path = export_yolo_model("yolov8n.pt", "onnx")
print(f"Exported model path: {onnx_path}")

pt_size = os.path.getsize("yolov8n.pt") / (1024 * 1024)
onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"PyTorch model size: {pt_size:.2f} MB")
print(f"ONNX model size: {onnx_size:.2f} MB")

# Benchmarking latency
dummy_input = torch.randn(1, 3, 640, 640)
model_pt = YOLO("yolov8n.pt")

t0 = time.time()
for _ in range(10):
    model_pt.predict(dummy_input, verbose=False)
pt_latency = (time.time() - t0) * 1000 / 10.0

model_onnx = YOLO(onnx_path)
t0 = time.time()
for _ in range(10):
    model_onnx.predict(dummy_input, verbose=False)
onnx_latency = (time.time() - t0) * 1000 / 10.0

print(f"PyTorch Average Inference Latency: {pt_latency:.2f} ms")
print(f"ONNX Average Inference Latency: {onnx_latency:.2f} ms")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Latency comparison
plt.figure(figsize=(6, 4))
plt.bar(["PyTorch", "ONNX"], [pt_latency, onnx_latency], color=['blue', 'green'])
plt.ylabel("Latency (ms)")
plt.title("Inference Latency: PyTorch vs. ONNX")
plt.show()"""
        }
    },
    "EX59_Multi_Task_Modes": {
        "en": {
            "title": "🧠 EX59: Multi-Task Modes",
            "markdown": """# 🧠 EX59: Multi-Task Modes

Ultralytics YOLO supports several computer vision tasks beyond standard object detection.

## 1. Supported Vision Tasks
- **Detect**: Classifies objects and finds bounding boxes (`yolo11n.pt`).
- **Segment**: Classifies, locates, and draws pixel-wise masks for objects (`yolo11n-seg.pt`).
- **Classify**: Assigns a label/category to the whole image (`yolo11n-cls.pt`).
- **Pose**: Estimates keypoints (e.g. joints, facial keypoints) for detected humans (`yolo11n-pose.pt`).
- **OBB (Oriented Bounding Box)**: Detects objects at custom rotated angles (`yolo11n-obb.pt`).

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import get_yolo_model_by_task
import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
tasks = ["detect", "segment", "classify", "pose"]
models = {}

# Load and audit models for each task
for task in tasks:
    print(f"Loading model for task: {task}")
    try:
        model = get_yolo_model_by_task(task, size="n")
        models[task] = model
        print(f"  Model name: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'YOLO11n'}")
        print(f"  Number of target classes: {len(model.names)}")
    except Exception as e:
        print(f"  Failed to load {task} model: {e}")

# Run random tensor validation
dummy_input = torch.randn(1, 3, 224, 224)
print("\\nAuditing output shapes:")
for name, model in models.items():
    try:
        res = model.predict(dummy_input, verbose=False)
        print(f"  Task '{name}' inference successful.")
        first_res = res[0]
        if name == "detect":
            print(f"    Boxes output shape: {first_res.boxes.shape if first_res.boxes is not None else 'None'}")
        elif name == "segment":
            print(f"    Masks shape: {first_res.masks.shape if first_res.masks is not None else 'None'}")
        elif name == "classify":
            print(f"    Probs shape: {first_res.probs.shape if first_res.probs is not None else 'None'}")
        elif name == "pose":
            print(f"    Keypoints shape: {first_res.keypoints.shape if first_res.keypoints is not None else 'None'}")
    except Exception as e:
        print(f"  Failed prediction for '{name}': {e}")
print("--- AUDIT & INSPECTION END ---")

# Plot target class sizes
plt.figure(figsize=(8, 4))
plt.bar(list(models.keys()), [len(models[t].names) for t in models.keys()], color=['purple', 'teal', 'blue', 'orange'])
plt.ylabel("Number of Classes")
plt.title("Class Count Across Vision Tasks")
plt.show()"""
        },
        "th": {
            "title": "🧠 EX59: Multi-Task Modes (รูปแบบโครงงานหลากหลายประเภทร่วมกับ YOLO)",
            "markdown": """# 🧠 EX59: Multi-Task Modes (รูปแบบโครงงานหลากหลายประเภทร่วมกับ YOLO)

นอกจากงานตรวจจับวัตถุทั่วไปแล้ว Ultralytics YOLO ยังถูกสร้างขึ้นมาเพื่อให้รองรับกับประเภทงานทางคอมพิวเตอร์วิทัศน์ที่หลากหลายแบบครบวงจร

## 1. ประเภทงานที่รองรับ
- **Detect**: ตรวจจับประเภทพร้อมตีกรอบกรอบล้อมรอบวัตถุ (`yolo11n.pt`)
- **Segment**: ถอดภาพวัตถุออกทีละพิกเซลแบบแบ่งส่วนรายละเอียด (`yolo11n-seg.pt`)
- **Classify**: จำแนกประเภทสิ่งของที่ปรากฏในภาพโดยรวม (`yolo11n-cls.pt`)
- **Pose**: วิเคราะห์และตรวจจับท่าทางและโครงกระดูกของร่างกาย (`yolo11n-pose.pt`)
- **OBB (Oriented Bounding Box)**: ค้นหาวัตถุที่มีการหักเหและหมุนเอียง (`yolo11n-obb.pt`)

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import get_yolo_model_by_task
import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
tasks = ["detect", "segment", "classify", "pose"]
models = {}

# Load and audit models for each task
for task in tasks:
    print(f"Loading model for task: {task}")
    try:
        model = get_yolo_model_by_task(task, size="n")
        models[task] = model
        print(f"  Model name: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'YOLO11n'}")
        print(f"  Number of target classes: {len(model.names)}")
    except Exception as e:
        print(f"  Failed to load {task} model: {e}")

# Run random tensor validation
dummy_input = torch.randn(1, 3, 224, 224)
print("\\nAuditing output shapes:")
for name, model in models.items():
    try:
        res = model.predict(dummy_input, verbose=False)
        print(f"  Task '{name}' inference successful.")
        first_res = res[0]
        if name == "detect":
            print(f"    Boxes output shape: {first_res.boxes.shape if first_res.boxes is not None else 'None'}")
        elif name == "segment":
            print(f"    Masks shape: {first_res.masks.shape if first_res.masks is not None else 'None'}")
        elif name == "classify":
            print(f"    Probs shape: {first_res.probs.shape if first_res.probs is not None else 'None'}")
        elif name == "pose":
            print(f"    Keypoints shape: {first_res.keypoints.shape if first_res.keypoints is not None else 'None'}")
    except Exception as e:
        print(f"  Failed prediction for '{name}': {e}")
print("--- AUDIT & INSPECTION END ---")

# Plot target class sizes
plt.figure(figsize=(8, 4))
plt.bar(list(models.keys()), [len(models[t].names) for t in models.keys()], color=['purple', 'teal', 'blue', 'orange'])
plt.ylabel("Number of Classes")
plt.title("Class Count Across Vision Tasks")
plt.show()"""
        }
    },
    "EX60_Streaming_Inference": {
        "en": {
            "title": "🧠 EX60: Streaming Inference",
            "markdown": """# 🧠 EX60: Streaming Inference

Processing long video sequences frame-by-frame can easily consume system memory if all intermediate predictions are saved.

## 1. Python Generator and Streaming
- **`stream=True`**: Standard prediction runs in a memory-safe generator mode. This maintains $O(1)$ space complexity relative to video length, discarding processed images and boxes after yielding.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import stream_inference_generator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

# Create a mock video file (15 frames)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = "stream_demo.mp4"
out = cv2.VideoWriter(video_path, fourcc, 15.0, (320, 240))
for i in range(15):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(img, (80 + i * 10, 120), 20, (255, 255, 255), -1)
    out.write(img)
out.release()

print("--- AUDIT & INSPECTION START ---")
print("Streaming inference over video...")
gen = stream_inference_generator("yolov8n.pt", video_path)

frame_detections = []
for idx, detections in enumerate(gen):
    print(f"Frame {idx}: Detected {len(detections)} objects.")
    frame_detections.append(len(detections))
    if detections:
        print(f"  First object box: {detections[0]['box']}")
        print(f"  Confidence: {detections[0]['confidence']:.4f}")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Plot detections per frame
plt.figure(figsize=(6, 4))
plt.plot(range(len(frame_detections)), frame_detections, marker='o', color='purple')
plt.xlabel("Frame Index")
plt.ylabel("Number of Detections")
plt.title("Object Detections over Video Frames")
plt.grid(True)
plt.show()

# Cleanup
if os.path.exists(video_path):
    os.remove(video_path)"""
        },
        "th": {
            "title": "🧠 EX60: Streaming Inference (การทำนายผลแบบใช้หน่วยความจำต่ำ)",
            "markdown": """# 🧠 EX60: Streaming Inference (การทำนายผลแบบใช้หน่วยความจำต่ำ)

การอ่านไฟล์และตรวจจับเฟรมวิดีโอที่มีความยาวมาก ๆ มีความเสี่ยงที่จะเกิดภาวะหน่วยความจำล้น (Memory leak/overflow) หากต้องเก็บประวัติเฟรมทั้งหมดไว้ในแรม

## 1. ยุทธวิธีของสตรีมมิ่งผ่านคำสั่ง Generator
- **`stream=True`**: เป็นการสั่งประมวลผลโมเดลที่ส่งมอบพิกัดทีละภาพโดยรักษาพิกัดความต้องการพื้นที่ของหน่วยความจำให้อยู่ในระดับต่ำสุด ($O(1)$) เมื่อดึงข้อมูลออกไปแล้ว เฟรมก่อนหน้าจะถูกลบออกจากหน่วยความจำทันที

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import stream_inference_generator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

# Create a mock video file (15 frames)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = "stream_demo.mp4"
out = cv2.VideoWriter(video_path, fourcc, 15.0, (320, 240))
for i in range(15):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(img, (80 + i * 10, 120), 20, (255, 255, 255), -1)
    out.write(img)
out.release()

print("--- AUDIT & INSPECTION START ---")
print("Streaming inference over video...")
gen = stream_inference_generator("yolov8n.pt", video_path)

frame_detections = []
for idx, detections in enumerate(gen):
    print(f"Frame {idx}: Detected {len(detections)} objects.")
    frame_detections.append(len(detections))
    if detections:
        print(f"  First object box: {detections[0]['box']}")
        print(f"  Confidence: {detections[0]['confidence']:.4f}")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Plot detections per frame
plt.figure(figsize=(6, 4))
plt.plot(range(len(frame_detections)), frame_detections, marker='o', color='purple')
plt.xlabel("Frame Index")
plt.ylabel("Number of Detections")
plt.title("Object Detections over Video Frames")
plt.grid(True)
plt.show()

# Cleanup
if os.path.exists(video_path):
    os.remove(video_path)"""
        }
    },
    "EX61_YOLO_Dataset_Format": {
        "en": {
            "title": "🧠 EX61: YOLO Dataset Format and Normalized Coordinate Conversion",
            "markdown": """# 🧠 EX61: YOLO Dataset Format and Normalized Coordinate Conversion

This notebook explains how bounding box coordinates are converted from raw pixel values to YOLO's normalized coordinate format.

## 1. Concept of YOLO Bounding Box Format
YOLO expects annotations in a `.txt` file corresponding to each image. Each line contains:
`<class_id> <x_center> <y_center> <width> <height>`

- **`class_id`**: Zero-indexed class ID (integer).
- **`x_center`, `y_center`**: Bounding box center coordinates, normalized by the image's width and height.
- **`width`, `height`**: Bounding box width and height, normalized by the image's width and height.

All coordinates are floats in the range $[0.0, 1.0]$.

## 2. Mathematical Equations
Suppose raw bounding box coordinates are given in $[x_{\\min}, y_{\\min}, x_{\\max}, y_{\\max}]$ (pixel coordinates), and the image has width $W$ and height $H$.

$$\\text{box\\_width} = x_{\\max} - x_{\\min}$$
$$\\text{box\\_height} = y_{\\max} - y_{\\min}$$
$$x_{\\text{center}} = \\frac{x_{\\min} + x_{\\max}}{2W}$$
$$y_{\\text{center}} = \\frac{y_{\\min} + y_{\\max}}{2H}$$
$$\\text{normalized\\_width} = \\frac{\\text{box\\_width}}{W}$$
$$\\text{normalized\\_height} = \\frac{\\text{box\\_height}}{H}$$

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import convert_to_yolo_format
import matplotlib.pyplot as plt
%matplotlib inline

# Inputs
bbox = [120, 160, 360, 400] # [xmin, ymin, xmax, ymax] in pixels
W, H = 640, 480
class_id = 1

print("--- AUDIT & INSPECTION START ---")
result_str = convert_to_yolo_format(bbox, W, H, class_id)
print(f"YOLO format output string: {result_str}")

# Audit conversion ranges
parts = result_str.split()
c_id, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
print("Auditing Coordinate ranges:")
print(f"  Class ID: {c_id}")
print(f"  Center X: {xc:.6f} (expected within [0, 1])")
print(f"  Center Y: {yc:.6f} (expected within [0, 1])")
print(f"  Width:    {w:.6f} (expected within [0, 1])")
print(f"  Height:   {h:.6f} (expected within [0, 1])")
print("--- AUDIT & INSPECTION END ---")

# Visualization of bounding boxes on coordinate space
plt.figure(figsize=(8, 6))
plt.xlim(0, W)
plt.ylim(H, 0) # Flip y-axis to match image pixel coordinate system

# Draw full image box
plt.gca().add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor='blue', linestyle='--', linewidth=1.5, label='Image Boundary'))
# Draw bounding box
plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=True, facecolor='red', alpha=0.3, label='Bounding Box'))
# Draw center point
plt.plot(xc * W, yc * H, 'go', label='Normalized Center')

plt.title("YOLO Bounding Box Format Conversion Visualization")
plt.xlabel("X pixels")
plt.ylabel("Y pixels")
plt.legend()
plt.grid(True)
plt.show()"""
        },
        "th": {
            "title": "🧠 EX61: รูปแบบของชุดข้อมูล YOLO และการแปลงพิกัดแบบนอร์มัลไลซ์",
            "markdown": """# 🧠 EX61: รูปแบบของชุดข้อมูล YOLO และการแปลงพิกัดแบบนอร์มัลไลซ์

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
$$x_{\\text{center}} = \\frac{x_{\\min} + x_{\\max}}{2W}$$
$$y_{\\text{center}} = \\frac{y_{\\min} + y_{\\max}}{2H}$$
$$\\text{normalized\\_width} = \\frac{\\text{box\\_width}}{W}$$
$$\\text{normalized\\_height} = \\frac{\\text{box\\_height}}{H}$$

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import convert_to_yolo_format
import matplotlib.pyplot as plt
%matplotlib inline

# Inputs
bbox = [120, 160, 360, 400] # [xmin, ymin, xmax, ymax] in pixels
W, H = 640, 480
class_id = 1

print("--- AUDIT & INSPECTION START ---")
result_str = convert_to_yolo_format(bbox, W, H, class_id)
print(f"YOLO format output string: {result_str}")

# Audit conversion ranges
parts = result_str.split()
c_id, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
print("Auditing Coordinate ranges:")
print(f"  Class ID: {c_id}")
print(f"  Center X: {xc:.6f} (expected within [0, 1])")
print(f"  Center Y: {yc:.6f} (expected within [0, 1])")
print(f"  Width:    {w:.6f} (expected within [0, 1])")
print(f"  Height:   {h:.6f} (expected within [0, 1])")
print("--- AUDIT & INSPECTION END ---")

# Visualization of bounding boxes on coordinate space
plt.figure(figsize=(8, 6))
plt.xlim(0, W)
plt.ylim(H, 0) # Flip y-axis to match image pixel coordinate system

# Draw full image box
plt.gca().add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor='blue', linestyle='--', linewidth=1.5, label='Image Boundary'))
# Draw bounding box
plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=True, facecolor='red', alpha=0.3, label='Bounding Box'))
# Draw center point
plt.plot(xc * W, yc * H, 'go', label='Normalized Center')

plt.title("YOLO Bounding Box Format Conversion Visualization")
plt.xlabel("X pixels")
plt.ylabel("Y pixels")
plt.legend()
plt.grid(True)
plt.show()"""
        }
    },
    "EX62_Inference_Visualization": {
        "en": {
            "title": "🧠 EX62: Inference and Prediction Visualization",
            "markdown": """# 🧠 EX62: Inference and Prediction Visualization

This notebook demonstrates how to run object detection inference using Ultralytics YOLO, visualize predictions using the built-in `plot()` method, and save the resulting image.

## 1. Concept of Prediction Plotting
The `plot()` method on the `Results` object automatically:
- Draws bounding boxes around detected objects.
- Places class names and confidence labels next to boxes.
- Returns the annotated image as a BGR numpy array (`numpy.ndarray`).

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import visualize_and_save
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

# Create a test source image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (200, 150), (450, 350), (0, 255, 255), -1) # Yellow box
cv2.imwrite("vis_input.jpg", img)

print("--- AUDIT & INSPECTION START ---")
print("Running inference and saving output...")
output_path = "vis_output.jpg"
visualize_and_save("yolov8n.pt", "vis_input.jpg", output_path)

if os.path.exists(output_path):
    print("Output file saved successfully.")
    # Audit saved image details
    out_img = cv2.imread(output_path)
    print(f"Saved Image dimensions: {out_img.shape}")
    print(f"Unique pixel values count: {len(np.unique(out_img))}")
else:
    print("Output file could not be found.")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Show result
if os.path.exists(output_path):
    img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title("Visualized Prediction Output")
    plt.axis("off")
    plt.show()

# Cleanup
if os.path.exists("vis_input.jpg"):
    os.remove("vis_input.jpg")
if os.path.exists("vis_output.jpg"):
    os.remove("vis_output.jpg")"""
        },
        "th": {
            "title": "🧠 EX62: การทำนายผลของ YOLO และการแสดงผลลัพธ์พิกัด",
            "markdown": """# 🧠 EX62: การทำนายผลของ YOLO และการแสดงผลลัพธ์พิกัด

สมุดบันทึกนี้แสดงวิธีการเรียกใช้การตรวจจับวัตถุโดยใช้ Ultralytics YOLO, แสดงผลลัพธ์การทำนายโดยใช้เมธอด `plot()` ในตัว และบันทึกรูปภาพผลลัพธ์

## 1. แนวคิดของการพล็อตผลลัพธ์การทำนาย
เมธอด `plot()` บนวัตถุ `Results` จะทำงานดังนี้:
- วาดกรอบล้อมรอบวัตถุที่ตรวจพบ
- วางป้ายกำกับชื่อคลาสและคะแนนความมั่นใจข้างกรอบ
- ส่งคืนภาพที่เขียนคำอธิบายประกอบแล้วในรูปแบบอาเรย์ BGR numpy (`numpy.ndarray`)

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import visualize_and_save
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

# Create a test source image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (200, 150), (450, 350), (0, 255, 255), -1) # Yellow box
cv2.imwrite("vis_input.jpg", img)

print("--- AUDIT & INSPECTION START ---")
print("Running inference and saving output...")
output_path = "vis_output.jpg"
visualize_and_save("yolov8n.pt", "vis_input.jpg", output_path)

if os.path.exists(output_path):
    print("Output file saved successfully.")
    # Audit saved image details
    out_img = cv2.imread(output_path)
    print(f"Saved Image dimensions: {out_img.shape}")
    print(f"Unique pixel values count: {len(np.unique(out_img))}")
else:
    print("Output file could not be found.")
print("--- AUDIT & INSPECTION END ---")

# Visualization: Show result
if os.path.exists(output_path):
    img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title("Visualized Prediction Output")
    plt.axis("off")
    plt.show()

# Cleanup
if os.path.exists("vis_input.jpg"):
    os.remove("vis_input.jpg")
if os.path.exists("vis_output.jpg"):
    os.remove("vis_output.jpg")"""
        }
    },
    "EX63_Multi_Object_Tracking": {
        "en": {
            "title": "🧠 EX63: Multi-Object Tracking (MOT) in Video Sequences",
            "markdown": """# 🧠 EX63: Multi-Object Tracking (MOT) in Video Sequences

This notebook covers how to trace object trajectories over consecutive frames using YOLO's tracking API.

## 1. Tracking Concept
While object detection processes each frame independently, tracking links detections across frames using motion dynamics and appearance features to assign unique, persistent track IDs (e.g. BoT-SORT, ByteTrack).

## 2. Extracting Track IDs
Inside the `Results` object, `results[0].boxes.id` contains the unique tracking ID tensor. Since some frames may have no active tracks, checking for `None` before parsing is necessary.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import track_objects_in_video
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

# Create a moving circle video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = "tracking_demo.mp4"
out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
for i in range(10):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (150 + i * 30, 240), 40, (255, 255, 255), -1)
    out.write(img)
out.release()

print("--- AUDIT & INSPECTION START ---")
print("Running tracking...")
track_ids = track_objects_in_video("yolov8n.pt", video_path)
print(f"Assigned track IDs: {track_ids}")
print("--- AUDIT & INSPECTION END ---")

# Plotting tracked trajectory
plt.figure(figsize=(6, 4))
x_traj = [150 + i * 30 for i in range(10)]
y_traj = [240 for _ in range(10)]
plt.plot(x_traj, y_traj, 'ro-', label='Object Trajectory')
plt.xlim(0, 640)
plt.ylim(480, 0)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Object Trajectory Over Time (Track ID 1)")
plt.legend()
plt.grid(True)
plt.show()

# Cleanup
if os.path.exists(video_path):
    os.remove(video_path)"""
        },
        "th": {
            "title": "🧠 EX63: การติดตามวัตถุหลายชิ้น (Multi-Object Tracking) ในลำดับวิดีโอ",
            "markdown": """# 🧠 EX63: การติดตามวัตถุหลายชิ้น (Multi-Object Tracking) ในลำดับวิดีโอ

สมุดบันทึกนี้ครอบคลุมวิธีการติดตามเส้นทางการเคลื่อนที่ของวัตถุข้ามเฟรมต่อเนื่องโดยใช้ API การติดตามของ YOLO

## 1. แนวคิดของการติดตามวัตถุ (Tracking Concept)
ในขณะที่การตรวจจับวัตถุทั่วไปประมวลผลแต่ละเฟรมอย่างเป็นอิสระ การติดตามวัตถุจะเชื่อมโยงผลการตรวจจับข้ามเฟรมต่างๆ โดยใช้การเคลื่อนที่ทางพลศาสตร์และคุณลักษณะของภาพภายนอก เพื่อกำหนดรหัสติดตาม (Track ID) ที่สม่ำเสมอและไม่ซ้ำกัน (เช่น BoT-SORT หรือ ByteTrack)

## 2. การดึงข้อมูล Track ID
ภายในวัตถุ `Results` ตัวแปร `results[0].boxes.id` จะเก็บเทนเซอร์รหัสติดตามที่ไม่ซ้ำกัน เนื่องจากบางเฟรมอาจไม่มีวัตถุที่กำลังติดตามอยู่ จึงจำเป็นต้องตรวจสอบค่า `None` ก่อนนำข้อมูลไปประมวลผล

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from solution import track_objects_in_video
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

# Create a moving circle video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = "tracking_demo.mp4"
out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
for i in range(10):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (150 + i * 30, 240), 40, (255, 255, 255), -1)
    out.write(img)
out.release()

print("--- AUDIT & INSPECTION START ---")
print("Running tracking...")
track_ids = track_objects_in_video("yolov8n.pt", video_path)
print(f"Assigned track IDs: {track_ids}")
print("--- AUDIT & INSPECTION END ---")

# Plotting tracked trajectory
plt.figure(figsize=(6, 4))
x_traj = [150 + i * 30 for i in range(10)]
y_traj = [240 for _ in range(10)]
plt.plot(x_traj, y_traj, 'ro-', label='Object Trajectory')
plt.xlim(0, 640)
plt.ylim(480, 0)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Object Trajectory Over Time (Track ID 1)")
plt.legend()
plt.grid(True)
plt.show()

# Cleanup
if os.path.exists(video_path):
    os.remove(video_path)"""
        }
    },
    "EX64_Callbacks_and_Logging": {
        "en": {
            "title": "🧠 EX64: Callback Hooks and Custom Metric Logging in YOLO",
            "markdown": """# 🧠 EX64: Callback Hooks and Custom Metric Logging in YOLO

This notebook demonstrates how to extend YOLO training and validation using custom event callbacks.

## 1. The Callbacks Architecture
Ultralytics YOLO uses a callback system to trigger user-defined functions at key events (e.g., `on_train_start`, `on_val_end`, `on_fit_epoch_end`). This decouples training control and logging logic from core network code.

## 2. Implementing Callback Functions
A callback function registered under validation events (like `on_val_end`) receives the `validator` object as its argument.
- The validation metrics are retrieved using `validator.metrics.results_dict`.
- Bounding box metrics keys are formatted as `"metrics/mAP50(B)"` and `"metrics/mAP50-95(B)"`.

## 🔗 Learning Integration
- For the full roadmap, see [[YOLO_Learning_Plan]].
- Document your progress and reflections in the [[learning_journal]].""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from ultralytics import YOLO
from solution import register_custom_metric_callback
import matplotlib.pyplot as plt
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
model = YOLO("yolov8n.pt")
print(f"Pre-callback list for 'on_val_end': {model.callbacks.get('on_val_end', [])}")

model = register_custom_metric_callback(model)
callbacks = model.callbacks.get("on_val_end", [])
print(f"Post-callback list for 'on_val_end': {callbacks}")
print("--- AUDIT & INSPECTION END ---")

# Trigger callback using a short validation run on coco8.yaml
print("Running validation on coco8 dataset to trigger metrics hook...")
try:
    results = model.val(data="coco8.yaml", imgsz=320, device="cpu", verbose=False)
    mAP50 = results.box.map50
    mAP95 = results.box.map
    
    # Plotting validation results
    plt.figure(figsize=(6, 4))
    plt.bar(["mAP@0.5", "mAP@0.5:0.95"], [mAP50, mAP95], color=['green', 'darkgreen'])
    plt.ylabel("Accuracy Score")
    plt.title("Validation Results Triggered via Callback")
    plt.ylim(0, 1.0)
    for i, val in enumerate([mAP50, mAP95]):
        plt.text(i, val + 0.02, f"{val:.3f}", ha='center')
    plt.show()
except Exception as e:
    print(f"Validation hook failed or skipped: {e}")"""
        },
        "th": {
            "title": "🧠 EX64: ระบบคอลแบ็ก (Callback Hooks) และการบันทึกมาตรวัดแบบกำหนดเองใน YOLO",
            "markdown": """# 🧠 EX64: ระบบคอลแบ็ก (Callback Hooks) และการบันทึกมาตรวัดแบบกำหนดเองใน YOLO

สมุดบันทึกนี้แสดงวิธีการขยายขีดความสามารถการฝึกฝนและการตรวจสอบความถูกต้องของ YOLO โดยใช้เหตุการณ์คอลแบ็กแบบกำหนดเอง

## 1. สถาปัตยกรรมคอลแบ็ก (Callbacks Architecture)
Ultralytics YOLO ใช้ระบบคอลแบ็กเพื่อเรียกใช้งานฟังก์ชันที่กำหนดโดยผู้ใช้ในเหตุการณ์สำคัญต่างๆ (เช่น `on_train_start`, `on_val_end`, `on_fit_epoch_end`) สิ่งนี้ช่วยแยกตรรกะการควบคุมและการบันทึกข้อมูลออกจากการทำงานของโครงข่ายประสาทหลัก

## 2. การสร้างฟังก์ชันคอลแบ็ก
ฟังก์ชันคอลแบ็กที่ลงทะเบียนในเหตุการณ์การวัดผล (เช่น `on_val_end`) จะรับวัตถุ `validator` เป็นอาร์กิวเมนต์
- มาตรวัดความถูกต้องจะถูกดึงออกมาโดยใช้ `validator.metrics.results_dict`
- คีย์สำหรับมาตรวัดระดับกรอบล้อมรอบ (Bounding Box) จะอยู่ในรูปแบบ `"metrics/mAP50(B)"` และ `"metrics/mAP50-95(B)"`

## 🔗 ลิงก์บันทึกการเรียนรู้
- สำหรับแผนการเรียนรู้หลัก ดูที่ [[YOLO_Learning_Plan]]
- จดบันทึกและทบทวนความรู้ของคุณได้ใน [[learning_journal]]""",
            "code": """# Back up or checkpoint this section of code before starting to modify the large file.
from ultralytics import YOLO
from solution import register_custom_metric_callback
import matplotlib.pyplot as plt
%matplotlib inline

print("--- AUDIT & INSPECTION START ---")
model = YOLO("yolov8n.pt")
print(f"Pre-callback list for 'on_val_end': {model.callbacks.get('on_val_end', [])}")

model = register_custom_metric_callback(model)
callbacks = model.callbacks.get("on_val_end", [])
print(f"Post-callback list for 'on_val_end': {callbacks}")
print("--- AUDIT & INSPECTION END ---")

# Trigger callback using a short validation run on coco8.yaml
print("Running validation on coco8 dataset to trigger metrics hook...")
try:
    results = model.val(data="coco8.yaml", imgsz=320, device="cpu", verbose=False)
    mAP50 = results.box.map50
    mAP95 = results.box.map
    
    # Plotting validation results
    plt.figure(figsize=(6, 4))
    plt.bar(["mAP@0.5", "mAP@0.5:0.95"], [mAP50, mAP95], color=['green', 'darkgreen'])
    plt.ylabel("Accuracy Score")
    plt.title("Validation Results Triggered via Callback")
    plt.ylim(0, 1.0)
    for i, val in enumerate([mAP50, mAP95]):
        plt.text(i, val + 0.02, f"{val:.3f}", ha='center')
    plt.show()
except Exception as e:
    print(f"Validation hook failed or skipped: {e}")"""
        }
    }
}

modified_files = []

for ex_num in range(51, 65):
    folder_name = None
    if ex_num == 51:
        folder_name = "EX51_YOLO_CLI"
    elif ex_num == 52:
        folder_name = "EX52_Model_Configurations"
    elif ex_num == 53:
        folder_name = "EX53_Training_Settings"
    elif ex_num == 54:
        folder_name = "EX54_Resuming_Training"
    elif ex_num == 55:
        folder_name = "EX55_Evaluation_Modes"
    elif ex_num == 56:
        folder_name = "EX56_Prediction_Sources"
    elif ex_num == 57:
        folder_name = "EX57_Bounding_Box_Parsing"
    elif ex_num == 58:
        folder_name = "EX58_Model_Export"
    elif ex_num == 59:
        folder_name = "EX59_Multi_Task_Modes"
    elif ex_num == 60:
        folder_name = "EX60_Streaming_Inference"
    elif ex_num == 61:
        folder_name = "EX61_YOLO_Dataset_Format"
    elif ex_num == 62:
        folder_name = "EX62_Inference_Visualization"
    elif ex_num == 63:
        folder_name = "EX63_Multi_Object_Tracking"
    elif ex_num == 64:
        folder_name = "EX64_Callbacks_and_Logging"
    
    if not folder_name:
        continue
        
    ex_path = base_dir / folder_name
    content_pair = notebooks_content.get(folder_name)
    
    for lang in ["en", "th"]:
        filename = "explanation.ipynb" if lang == "en" else "explanation_TH.ipynb"
        file_path = ex_path / filename
        
        # Make backup
        backup_path = ex_path / f"{filename}.bak"
        if file_path.exists():
            shutil.copy2(file_path, backup_path)
            print(f"Backed up: {file_path} -> {backup_path}")
        else:
            print(f"Skipping backup: {file_path} does not exist.")
            
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [line + "\n" for line in content_pair[lang]["markdown"].split("\n")]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + "\n" for line in content_pair[lang]["code"].split("\n")]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Updated: {file_path}")
        modified_files.append(str(file_path))

print("\nAll notebooks have been backed up and successfully rewritten!")
