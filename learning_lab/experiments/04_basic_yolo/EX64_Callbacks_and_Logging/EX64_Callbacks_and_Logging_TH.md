# 🧠 EX64: คอลแบ็กและการบันทึกข้อมูลการเรียนรู้ (Callbacks and Logging)

ในระหว่างการฝึกสอนโมเดล การตรวจติดตามตัวชี้วัด (metrics) เช่น ค่าความสูญเสีย (loss), ความแม่นยำ (precision), อัตราการตรวจจับได้จริง (recall), และค่า mAP ถือเป็นหัวใจสำคัญในการตรวจสอบความก้าวหน้าและการลู่เข้าหาค่าเหมาะสม (convergence) ของโมเดล โดย Ultralytics YOLO มีระบบเชื่อมโยงอัตโนมัติกับแพลตฟอร์มเก็บบันทึกข้อมูล (loggers) ยอดนิยม และจัดเตรียมระบบคอลแบ็ก (callbacks) เพื่อให้คุณเขียนโค้ดสั่งการทำงานเพิ่มเติม ณ ช่วงเวลาสำคัญต่างๆ ระหว่างการฝึกสอนโมเดล

---

## 1. การบันทึกข้อมูลเริ่มต้นและโครงสร้างไดเรกทอรี (Default Logging)

ทุกครั้งที่คุณสั่งฝึกสอนโมเดล YOLO จะสร้างไดเรกทอรีบันทึกผลการทำงานให้อัตโนมัติ (เช่น `runs/detect/train/`) ซึ่งมีโครงสร้างดังนี้:
*   `weights/`: เก็บไฟล์โมเดลที่บันทึกระหว่างการสอน (`best.pt` และ `last.pt`)
*   `results.csv`: เก็บข้อมูลตัวชี้วัดประสิทธิภาพในแต่ละรอบการฝึกสอน (epoch)
*   `results.png` & `confusion_matrix.png`: แผนภูมิแสดงกราฟการเรียนรู้และตาราง Confusion Matrix เพื่อวัดความผิดพลาดในการจำแนกประเภท
*   `val_batch0_labels.jpg`: รูปภาพจำลองของตัวอย่างข้อมูลที่ใช้สำหรับการฝึกสอนและทดสอบความถูกต้อง เพื่อช่วยตรวจสอบความถูกต้องของการทำ Augmentation และการระบุป้ายกำกับ

### การเชื่อมต่อกับแพลตฟอร์มเก็บบันทึก (Logging Integrations)
หากระบบคอมพิวเตอร์ของคุณมีการติดตั้งไลบรารีดังต่อไปนี้ YOLO จะทำการเปิดระบบและส่งประวัติการเรียนรู้ของโมเดลขึ้นสู่แพลตฟอร์มเหล่านั้นโดยอัตโนมัติโดยไม่ต้องตั้งค่าเพิ่มเติมใดๆ:
*   **TensorBoard**: เปิดใช้งานเริ่มต้นสำหรับเก็บข้อมูลในเครื่อง โดยบันทึกไฟล์ล็อกไว้ในไดเรกทอรีรันการทดสอบ คุณสามารถเรียกใช้งานผ่านคำสั่ง `tensorboard --logdir runs/` เพื่อเปิดดูผ่านหน้าเว็บ
*   **Weights & Biases (W&B)**: ตรวจจับและอัปโหลดประวัติไปเก็บไว้บนคลาวด์ พร้อมแสดงการเปรียบเทียบ สามารถเปิดใช้งานได้ง่ายๆ ด้วยคำสั่ง `pip install wandb`
*   **MLflow / Comet / ClearML**: เปิดใช้งานเชื่อมต่อโดยอัตโนมัติทันทีที่ตรวจพบแพ็กเกจเหล่านี้ในสภาพแวดล้อม Python และมีการตั้งค่าตัวแปรระบบ (environment variables) ไว้เรียบร้อย

---

## 2. การเขียนฟังก์ชันคอลแบ็กแบบกำหนดเอง (Writing Custom Callback Functions)

คอลแบ็ก (Callbacks) คือ ฟังก์ชันที่จะถูกเรียกทำงานโดยอัตโนมัติเมื่อสิ้นสุดช่วงเวลาหรือเหตุการณ์เฉพาะในลูปฝึกสอน ทดสอบ หรือทำนายผล คุณสามารถนำไปประยุกต์ใช้งานเพื่อดึงข้อมูลประสิทธิภาพ, ส่งข้อความแจ้งเตือนผ่าน Telegram, ปรับอัตราการเรียนรู้ (learning rates), หรือส่งข้อมูลไปยังระบบเก็บบันทึกอื่นๆ ที่ YOLO ไม่ได้รองรับอย่างเป็นทางการ

เหตุการณ์สำคัญ (event hooks) ที่ YOLO มีมาให้ใช้งานประกอบด้วย:
*   `on_train_start`: ทำงานเมื่อโมเดลเริ่มกระบวนการฝึกสอน
*   `on_train_epoch_end`: ทำงานเมื่อสิ้นสุดรอบการสอน (epoch) ของข้อมูลฝึกสอน
*   `on_fit_epoch_end`: ทำงานหลังจากประเมินความแม่นยำ (validation) บนรอบนั้นๆ เสร็จสิ้น
*   `on_train_end`: ทำงานเมื่อฝึกสอนครบจำนวนรอบและกระบวนการทั้งหมดเสร็จสิ้น

### โครงสร้างของฟังก์ชันคอลแบ็ก
ฟังก์ชันคอลแบ็กจำเป็นต้องกำหนดพารามิเตอร์รับออบเจกต์ `trainer` เสมอ ซึ่งตัวแปร `trainer` นี้จะช่วยให้เราสามารถเข้าถึงข้อมูลของระบบ ณ ขณะนั้นได้ เช่น:
*   `trainer.epoch`: ลำดับรอบปัจจุบัน (เริ่มนับจากดัชนี $0$)
*   `trainer.epochs`: จำนวนรอบการเรียนรู้ทั้งหมดที่กำหนดไว้
*   `trainer.loss_items`: ค่าความสูญเสียสะสมในรอบปัจจุบัน
*   `trainer.metrics`: พจนานุกรมรวบรวมตัวชี้วัดคะแนนความถูกต้องจากการประเมินผล (เช่น `fitness`, `mAP50`, `mAP50-95`)

---

## 3. ตัวอย่างการเขียนโค้ด Python

โค้ดด้านล่างนี้คือตัวอย่างการสร้างฟังก์ชันคอลแบ็กเพื่อพิมพ์รายงานสรุปผลลัพธ์หลังจบรอบการตรวจวัด พร้อมทั้งทำการลงทะเบียนเข้าสู่ระบบและสั่งฝึกสอนโมเดล:

```python
from ultralytics import YOLO

# 1. Define custom callback function
def log_epoch_summary(trainer):
    """
    A custom callback triggered at the end of each fit epoch.
    Prints a customized message showing loss and fitness metrics.
    """
    epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
    total_epochs = trainer.epochs
    
    # Extract loss values
    # loss_items returns a list of loss values (e.g., Box loss, Class loss, DFL loss)
    loss_val = trainer.loss_items.tolist()
    
    # Extract validation metrics
    # metrics is a dictionary containing validation performance
    metrics = trainer.metrics
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    
    print(f"\n=========================================")
    print(f"🎓 Professor Monitor - Epoch [{epoch}/{total_epochs}]")
    print(f"--> Current Box/Class/DFL Losses: {loss_val}")
    print(f"--> Validation mAP50-95: {map50_95:.4f}")
    print(f"=========================================\n")

# 2. Load the model
model = YOLO("yolo11n.pt")

# 3. Register the callback to the 'on_fit_epoch_end' hook
# Note: You register callbacks BEFORE calling model.train()
model.add_callback("on_fit_epoch_end", log_epoch_summary)

# 4. Start training
# The callback will execute automatically at the end of each epoch
model.train(
    data="coco8.yaml",
    epochs=3,
    imgsz=640,
    device="cpu"  # Change to 0 if GPU is available
)
```

---

## 💡 คำแนะนำจากอาจารย์ (Professor Tips)

1.  **การเขียนทับคอลแบ็กเริ่มต้น (Overwriting Default Callbacks)**: ตามปกติ YOLO จะกำหนดคอลแบ็กสำหรับ TensorBoard และ W&B ไว้เป็นมาตรฐาน หากคุณต้องการล้างระบบคอลแบ็กเริ่มต้นทั้งหมดเพื่อป้องกันข้อมูลล็อกซ้ำซ้อน คุณสามารถสั่งลบลิสต์คอลแบ็กภายในพจนานุกรมประตัวได้ด้วยคำสั่ง:
    ```python
    model.callbacks["on_fit_epoch_end"] = []
    ```
    หลังจากนั้นจึงค่อยใช้เมธอด `.add_callback()` เพิ่มคอลแบ็กเฉพาะของคุณเข้าไปใหม่
2.  **การดึงข้อมูลสถานะฮาร์ดแวร์ GPU**: ภายในฟังก์ชันคอลแบ็ก คุณสามารถเขียนคำสั่งตรวจสอบความร้อนและการทำงานของ GPU ได้ด้วยการทำงานร่วมกับไลบรารี `psutil` หรือ `torch.cuda` ทำให้สามารถเขียนโค้ดแจ้งเตือนหรือสั่งหยุดชั่วคราวเมื่อระบบร้อนเกินไปได้
3.  **ข้อแตกต่างระหว่าง `on_train_epoch_end` และ `on_fit_epoch_end`**: 
    *   `on_train_epoch_end` จะทำงานทันทีหลังจากโมเดลเรียนรู้จากข้อมูลฝึกสอนในรอบนั้นๆ ครบแล้ว แต่ *ก่อน* ที่จะเริ่มประเมินผลกับชุดข้อมูลตรวจสอบความถูกต้อง
    *   `on_fit_epoch_end` จะทำงาน *หลัง* จากโมเดลเรียนรู้และประเมินผลความแม่นยำบนรอบนั้นเสร็จสิ้นทั้งหมดแล้ว จึงเป็นจุดที่เหมาะสมที่สุดสำหรับการเรียกดูคะแนนความถูกต้องเช่น mAP

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX53_Training_Settings_TH|EX53: การตั้งค่าการฝึกสอน (Training Settings)]]
*   [[EX55_Evaluation_Modes_TH|EX55: โหมดการประเมินผลโมเดล (Evaluation Modes)]]
*   กลับสู่แผนการเรียนหลัก: [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal|บันทึกการเรียนรู้]]
