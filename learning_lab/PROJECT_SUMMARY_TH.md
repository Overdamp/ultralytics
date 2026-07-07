# 📊 การตรวจจับวัตถุด้วย YOLO: สรุปภาพรวมโครงการและโครงสร้างไดเรกทอรี (Project Summary & Directory Overview)

ยินดีต้อนรับสู่ **Ultralytics YOLO Learning Lab & โครงการตรวจจับวัตถุแบบกำหนดเอง (Custom Object Detection)**! ในฐานะอาจารย์ผู้สอนวิชาคอมพิวเตอร์วิทัศน์ (Computer Vision) ผมได้จัดทำเอกสารสรุปภาพรวมโครงการฉบับภาษาไทยนี้ขึ้นมาเพื่อช่วยให้คุณสำรวจพื้นที่ทำงาน เข้าใจทฤษฎีเบื้องหลังของเครือข่ายแบบไร้จุดยึด (Anchor-Free Networks) วิเคราะห์คุณลักษณะของชุดข้อมูล และทบทวนหลักสูตรการเรียนรู้ทั้งหมดของเรา

ไฟล์นี้จะทำหน้าที่เป็นโหนดทางเข้าหลัก (Root Entrypoint) สำหรับ Obsidian Knowledge Vault ของคุณ

---

## 🏗️ 1. ภาพรวมโครงการและวัตถุประสงค์ (Project Overview & Objective)

วัตถุประสงค์หลักของโครงการนี้คือการสร้าง, ปรับจูน (Fine-tuning), และนำไปใช้งานจริง (Deployment) ของโมเดล **YOLO** แบบกำหนดเอง (โดยเฉพาะสถาปัตยกรรมรุ่นน้ำหนักเบาอย่าง `yolo26n.pt`) ที่ได้รับการออกแบบมาเพื่อตรวจจับ **วัตถุในกลุ่มอุตสาหกรรม PTT จำนวน 26 คลาส** (เช่น วาล์ว เกจ หัวบ่อน้ำมัน และอุปกรณ์อุตสาหกรรมที่เกี่ยวข้อง)

คลังรหัสข้อมูล (Repository) นี้ทำหน้าที่เป็นทั้งพื้นที่ทำงานหลักและ **ห้องปฏิบัติการเรียนรู้คอมพิวเตอร์วิทัศน์ (Computer Vision Learning Lab)** ซึ่งจัดระเบียบในรูปแบบ Obsidian Vault ภายใต้ไดเรกทอรี `learning_lab/`

---

## 📈 2. การวิเคราะห์ชุดข้อมูล: ชุดข้อมูล PTT (PTT Dataset Analysis)

ชุดข้อมูลที่กำหนดเองของเราถูกตั้งค่าไว้ใน `datasets/coco8/overall-ptt-object-detection.v11i.yolov11/data.yaml` ซึ่งมีรายละเอียดทางสถิติดังนี้:

### A. ข้อมูลสถิติและการแบ่งส่วนชุดข้อมูล (Dataset Splits & Statistics)
*   **รูปภาพทั้งหมด (Total Images):** 5,211 ภาพ
*   **ส่วนฝึกสอน (Train Split):** 3,887 ภาพ (61,558 กล่องขอบเขต - Bounding Boxes)
*   **ส่วนตรวจสอบความถูกต้อง (Validation Split):** 773 ภาพ (12,309 กล่องขอบเขต)
*   **ส่วนทดสอบ (Test Split):** 551 ภาพ (8,462 กล่องขอบเขต)

### B. รายชื่อคลาสของวัตถุทั้ง 26 คลาส (Class Definitions)
ชุดข้อมูลนี้ประกอบด้วยคลาสที่มีความละเอียดสูง 26 คลาส ซึ่งแสดงถึงส่วนประกอบทางอุตสาหกรรม ข้อต่อท่อ อุปกรณ์ควบคุม และส่วนประกอบด้านความปลอดภัย:
1.  `actuator` (หัวขับวาล์ว)
2.  `analog-gauge` (เกจวัดแบบเข็ม)
3.  `control-valve` (วาล์วควบคุม)
4.  `control-valve-stem` (ก้านวาล์วควบคุม)
5.  `digital-gauge` (เกจวัดแบบดิจิทัล)
6.  `flange` (หน้าแปลน)
7.  `flow-fitting` (ข้อต่อท่อทางเดิน)
8.  `flow-line` (ท่อส่งไหล)
9.  `handwheel-handle` (พวงมาลัยหมุนวาล์ว)
10. `handwheel-valve` (วาล์วแบบพวงมาลัย)
11. `lever-handle` (ด้ามคันโยก)
12. `lever-valve` (วาล์วแบบคันโยก)
13. `manual-valve-stem` (ก้านวาล์วแมนนวล)
14. `meter` (มิเตอร์วัดค่า)
15. `pig-alert` (อุปกรณ์แจ้งเตือนการส่งลูกหมูทำความสะอาดท่อ)
16. `pig-closure` (ฝาปิดอุปกรณ์รับส่งลูกหมู)
17. `polished-rod` (ก้านสูบขัดมัน)
18. `positioner` (อุปกรณ์กำหนดตำแหน่งวาล์ว)
19. `pump` (ปั๊มน้ำมัน/ของเหลว)
20. `small-valve` (วาล์วขนาดเล็ก)
21. `spectacle-blind` (แผ่นกั้นท่อรูปแว่นตา)
22. `stuffing-box` (กล่องซีลกันรั่ว)
23. `valve-body` (ตัวเรือนวาล์ว)
24. `vertical-gauge` (เกจวัดแนวตั้ง)
25. `wellhead` (ปากบ่อน้ำมัน)
26. `xmas-tree` (ต้นคริสต์มาสควบคุมปากบ่อ)

> [!IMPORTANT]
> **การตรวจสอบความปลอดภัยทางข้อมูล:** ก่อนที่คุณจะเริ่มรันคำสั่งฝึกสอนใดๆ โปรดตรวจสอบเส้นทางไดเรกทอรีในไฟล์คอนฟิกูเรชันชุดข้อมูลของคุณที่ [data.yaml](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11/data.yaml) ทุกครั้งเพื่อป้องกันข้อผิดพลาดในการโหลดรูปภาพ นอกจากนี้ ผลการตรวจสอบพบว่ามีไฟล์ป้ายกำกับว่างเปล่า (Empty labels) 4 ไฟล์ในกลุ่มชุดฝึกสอน (เช่น `Part7-Drift_024355_jpg.rf...txt`) และพิกัดที่เสียหายเล็กน้อย แต่โดยภาพรวมการกระจายตัวของข้อมูลยังสมบูรณ์เพียงพอสำหรับการปรับจูนโมเดล

---

## 👨‍🏫 3. รากฐานทางทฤษฎีคอมพิวเตอร์วิทัศน์ (Theoretical Foundations)

ในการก้าวสู่การเป็นผู้เชี่ยวชาญด้านคอมพิวเตอร์วิทัศน์ คุณต้องเข้าใจกระบวนการทำงานของโมเดลอย่างลึกซึ้ง ไม่ใช่การมองระบบเป็นกล่องดำที่รันผ่านๆ:

### A. ความแตกต่างระหว่าง Anchor-Based และ Anchor-Free
โมเดล YOLO รุ่นเก่า (เช่น YOLOv3, YOLOv5) จะใช้ **Anchor Boxes** ซึ่งก็คือกล่องต้นแบบที่ระบุอัตราส่วนกว้างยาวไว้ล่วงหน้า จากนั้นเครือข่ายจะทำนายระยะห่าง (offset) เพื่อปรับเปลี่ยนกล่องต้นแบบเหล่านั้นให้ตรงกับวัตถุจริง
*   *ข้อจำกัด:* ต้องใช้ขั้นตอนคำนวณที่ซับซ้อน เช่น การทำ K-Means บนชุดข้อมูลเพื่อหากล่องต้นแบบที่เหมาะสมที่สุด และหากวัตถุจริงมีรูปร่างเปลี่ยนไปจากเดิมมาก โมเดลจะตรวจจับได้ยาก
*   *แบบไร้จุดยึด (Anchor-Free - YOLOv8/YOLO11/YOLO26):* โมเดลสมัยใหม่จะทำนายพิกัดจากจุดศูนย์กลางของตารางพิกเซลไปยังขอบทั้งสี่ด้านของวัตถุโดยตรง:
    $$\Delta x, \Delta y, \Delta w, \Delta h$$
    สถาปัตยกรรมแบบนี้ช่วยลดพารามิเตอร์การตั้งค่า (hyperparameters) ทำให้โมเดลรวบรวมข้อมูลได้ไวขึ้น และเข้ากับวัตถุขนาดเล็กหรือรูปทรงแปลกๆ ได้ดีกว่า

### B. ความเข้าใจเกี่ยวกับเมทริกซ์การประเมินผล (Evaluation Metrics)
การวัดประสิทธิภาพของโมเดลสำหรับการประเมินผลใช้ตัวชี้วัดหลักดังนี้:
*   **ความแม่นยำ (Precision - $P$):** สัดส่วนของวัตถุที่โมเดลทำนายว่าใช่และถูกต้องจริงๆ ต่อจำนวนวัตถุที่ทำนายทั้งหมด ค่าความแม่นยำที่สูงหมายความว่าโมเดลเดาสุ่มน้อยลง
    $$P = \frac{TP}{TP + FP}$$
*   **ความครอบคลุม (Recall - $R$):** สัดส่วนของวัตถุที่โมเดลตรวจเจอจริงต่อจำนวนวัตถุจริงทั้งหมดที่มีอยู่ในภาพ ค่าความครอบคลุมที่สูงหมายถึงโมเดลมองข้ามวัตถุน้อยลง
    $$R = \frac{TP}{TP + FN}$$
*   **mAP@0.5:** ค่าเฉลี่ยความแม่นยำรวม (mean Average Precision) ที่คำนวณบนระดับความทับซ้อน IoU (Intersection over Union) เท่ากับ 0.5 ใช้บ่งชี้ระดับการระบุตำแหน่งวัตถุเบื้องต้น
*   **mAP@0.5:0.95:** ค่าเฉลี่ยความแม่นยำรวมที่หาค่าเฉลี่ยซ้ำในทุกระดับความทับซ้อน IoU ตั้งแต่ 0.5 ไปจนถึง 0.95 (ขั้นละ 0.05) ซึ่งเป็นเกณฑ์มาตรฐานของชุดข้อมูลสากล COCO ที่ใช้ชี้วัดว่ากล่องที่ตรวจจับล้อมรอบวัตถุได้แนบชิดและแม่นยำเพียงใด

---

## 📚 4. แผนหลักสูตรห้องปฏิบัติการ (74 แบบฝึกหัด)

หลักสูตรภายใต้ `learning_lab/experiments/` แบ่งออกเป็น 5 โมดูลการเรียนรู้:

### 🧩 โมดูล 01: พื้นฐานการเรียนรู้ของเครื่องแบบดั้งเดิม (Classic Machine Learning)
สร้างรากฐานด้านการประมาณค่าทางสถิติและแบบจำลองจำแนกประเภท:
*   **การวิเคราะห์การถดถอยเชิงเส้นและพหุนาม (Linear & Polynomial Regression)** (`EX01` - `EX02`)
*   **การควบคุมความซับซ้อนของแบบจำลอง (Regularization)** (`EX03` - `EX04`: Ridge & Lasso)
*   **การจำแนกประเภท (Classification)** (`EX05` - `EX10`: Logistic Regression, KNN, Decision Trees, Random Forests, SVM, Naive Bayes)

### 📊 โมดูล 02: การประเมินผลและการปรับแต่งไฮเปอร์พารามิเตอร์ (Evaluation & Tuning)
ทำความเข้าใจเกี่ยวกับการออกแบบคณิตศาสตร์ของตัววัดผลและกลไกการปรับปรุงน้ำหนัก:
*   **การพัฒนาตัวชี้วัดประสิทธิภาพ** (`EX11` - `EX18`: Accuracy, Confusion Matrix, Precision, Recall, F1, ROC/AUC, PR Curves)
*   **การประเมินแบบไขว้และการวิเคราะห์ความคลาดเคลื่อน (Cross Validation & Bias-Variance)** (`EX19` - `EX20`)
*   **อัลกอริทึมการเพิ่มประสิทธิภาพและกำหนดการปรับอัตราการเรียนรู้ (Optimizers & Schedulers)** (`EX21` - `EX30`: โครงสร้างย่อยของ Gradient Descent, Momentum, RMSprop, Adam, Learning rate decay, Weight decay, Batch sizes)

### 🧠 โมดูล 03: โครงสร้างการเรียนรู้เชิงลึกและ YOLO (Deep Learning & YOLO Foundations)
แกะกล่องการสร้างเครือข่ายประสาทเทียมตั้งแต่ระดับเซลล์เดี่ยวไปจนถึงหัวตรวจจับภาพ:
*   **ส่วนประกอบหลักของโครงข่ายประสาท** (`EX31` - `EX37`: MLPs, Activation Functions, Backpropagation, CNN Convolutions, Pooling, Feature Maps)
*   **การเรียนรู้แบบถ่ายโอนและการปรับจูนแบบจำลอง (Transfer Learning & Fine-Tuning)** (`EX38` - `EX40`)
*   **โครงสร้างของโมเดลการตรวจจับ** (`EX41` - `EX50`: Bounding Boxes, IoU, Non-Maximum Suppression (NMS), Grid Assignments, mAP parsing, Genetic Algorithm tuning)

### ⚡ โมดูล 04: การใช้งานเครื่องมือ YOLO API & CLI ทั่วไป
ขั้นตอนปฏิบัติในการควบคุมเฟรมเวิร์กผ่านคำสั่งและสคริปต์ Python:
*   **อินเตอร์เฟซบรรทัดคำสั่ง YOLO** ([[experiments/04_basic_yolo/EX51_YOLO_CLI/EX51_YOLO_CLI_TH|EX51: YOLO CLI]])
*   **การตั้งค่าและฝึกสอนโมเดลที่กำหนดเอง** ([[experiments/04_basic_yolo/EX52_Model_Configurations/EX52_Model_Configurations_TH|EX52: Model Configs]], [[experiments/04_basic_yolo/EX53_Training_Settings/EX53_Training_Settings_TH|EX53: Training Settings]], [[experiments/04_basic_yolo/EX54_Resuming_Training/EX54_Resuming_Training_TH|EX54: Resuming Training]])
*   **สคริปต์วิเคราะห์และประมวลผล** ([[experiments/04_basic_yolo/EX55_Evaluation_Modes/EX55_Evaluation_Modes_TH|EX55]], [[experiments/04_basic_yolo/EX56_Prediction_Sources/EX56_Prediction_Sources_TH|EX56]], [[experiments/04_basic_yolo/EX57_Bounding_Box_Parsing/EX57_Bounding_Box_Parsing_TH|EX57]], [[experiments/04_basic_yolo/EX58_Model_Export/EX58_Model_Export_TH|EX58]])
*   **ส่วนขยายระบบงานขั้นสูง** ([[experiments/04_basic_yolo/EX59_Multi_Task_Modes/EX59_Multi_Task_Modes_TH|EX59: Multi-Task]], [[experiments/04_basic_yolo/EX60_Streaming_Inference/EX60_Streaming_Inference_TH|EX60: Streaming]], [[experiments/04_basic_yolo/EX61_YOLO_Dataset_Format/EX61_YOLO_Dataset_Format_TH|EX61]], [[experiments/04_basic_yolo/EX62_Inference_Visualization/EX62_Inference_Visualization_TH|EX62]], [[experiments/04_basic_yolo/EX63_Multi_Object_Tracking/EX63_Multi_Object_Tracking_TH|EX63]], [[experiments/04_basic_yolo/EX64_Callbacks_and_Logging/EX64_Callbacks_and_Logging_TH|EX64]])

### 🚀 โมดูล 05: กรณีศึกษาการประยุกต์ใช้งาน YOLO ขั้นสูง (Advanced Case Studies)
แนวทางประยุกต์ใช้โมเดลกับโจทย์อุตสาหกรรมในระบบการทำงานจริง:
*   **เครื่องมือวัดและเซนเซอร์** (`EX65` - `EX66`: ระบบอ่านค่าเกจเข็มอะนาล็อก, ระบบอ่านป้ายทะเบียน LPR)
*   **ความปลอดภัยและการเฝ้าระวัง** (`EX67` - `EX68`: การตรวจจับอุปกรณ์เซฟตี้ PPE, การทำแผนภาพความหนาแน่นของผู้คน)
*   **การควบคุมคุณภาพในกระบวนการผลิต** (`EX69` - `EX70`: การตรวจจับตำหนิแบบละเอียดด้วย SAHI, การตรวจสอบชั้นวางสินค้าค้าปลีก)
*   **ระบบหุ่นยนต์และวิเคราะห์ภาพกีฬา** (`EX71` - `EX74`: การติดตามลูกบอล, ภาพจำลองมุมมองแบบ Bird's Eye View สำหรับรถยนต์ขับเคลื่อนอัตโนมัติ, การจำแนกเกรดพืชผลทางการเกษตร, การตรวจจับไฟไหม้ด้วยเซนเซอร์จับความร้อน)

---

## 🔍 5. การทำงานร่วมกันกับระบบ Obsidian Knowledge Vault

คุณสามารถคลิกผ่านลิงก์วิกิต่อไปนี้เพื่อข้ามไปยังหน้าเนื้อหาและรายงานความคืบหน้าได้ทันที:
*   **แผนที่การเรียนรู้หลัก:** [[YOLO_Learning_Plan|แผนการเรียนรู้และขั้นตอนความคืบหน้าหลัก]]
*   **บันทึกการทดลองรายวัน:** [[Journal/learning_journal|สมุดบันทึกสรุปผล ผลลัพธ์ และการแก้บัก]]
*   **คลังแนวคิดเชิงลึก (Core Concepts):**
    *   [[Concepts/YOLO_Loss_Functions|ฟังก์ชันการสูญเสียของ YOLO (CIoU, DFL, และ BCE Loss)]]
    *   [[Concepts/Object_Detection_Tips|คำแนะนำการรับมือกับปัญหา Overfitting ในงานตรวจจับวัตถุ]]

---
*คำแนะนำเพิ่มเติม: แนะนำให้เปิดฟังก์ชัน **Graph View** ใน Obsidian เพื่อแสดงโครงข่ายการเชื่อมโยงความรู้ระหว่างแบบฝึกหัดแต่ละบท บันทึกส่วนตัว และสูตรคณิตศาสตร์หลักแบบกราฟิกที่สวยงาม!*
