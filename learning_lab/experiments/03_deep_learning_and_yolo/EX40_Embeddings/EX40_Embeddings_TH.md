# 🧠 EX40: เวกเตอร์ฝังตัวของคุณลักษณะในคอมพิวเตอร์วิทัศน์ (Feature Embeddings in Computer Vision)

เวกเตอร์ฝังตัว (Embedding) คือการแทนค่าข้อมูลมิติสูง (เช่น รูปภาพ หรือกรอบล้อมรอบ) ด้วยเวกเตอร์ที่มีมิติต่ำลง โดยคุณลักษณะที่มีความหมายเชิงความหมายคล้ายคลึงกัน (เช่น วาล์วสองประเภทที่ต่างกัน) จะถูกแปลงเป็นจุดที่อยู่ใกล้เคียงกันในปริภูมิเวกเตอร์ต่อเนื่องนี้

---

## 1. การลดมิติข้อมูลใน CNN (Dimensionality Reduction in CNNs)
ชิ้นส่วนภาพดิบ (Raw Image Patches) ประกอบด้วยพิกเซลจำนวนมหาศาล ตัวอย่างเช่น ภาพตัดเฉพาะ (Crop) ของวาล์วขนาดเล็ก $128 \times 128$ พิกเซลในระบบสี RGB ประกอบด้วย:
$$128 \times 128 \times 3 = 49,152 \text{ values}$$

การส่งผ่านภาพตัดเฉพาะนี้เข้าไปในโครงข่ายหลักของ CNN (เช่น CSPDarknet ใน YOLO) จะลดขนาดของภาพในเชิงพื้นที่ลง ในขณะเดียวกันก็สกัดคุณลักษณะเชิงความหมายออกมาเป็นเวกเตอร์ 1 มิติขนาดกะทัดรัด (เช่น ขนาด $128$, $256$ หรือ $512$ มิติ) เวกเตอร์ที่มีความหนาแน่นนี้เรียกว่า **เวกเตอร์ฝังตัวของคุณลักษณะ (Feature Embedding)**

---

## 2. การวัดความคล้ายคลึงกันทางคณิตศาสตร์ (Mathematical Similarity Metrics)

ในการเปรียบเทียบเวกเตอร์ฝังตัวสองตัว $\mathbf{u}$ และ $\mathbf{v}$ เราจะใช้การวัดระยะทางหรือความคล้ายคลึงกันดังนี้:

### A. ความคล้ายคลึงของโคไซน์ (Cosine Similarity)
ความคล้ายคลึงของโคไซน์จะวัดค่าโคไซน์ของมุมระหว่างเวกเตอร์สองตัว มีค่าตั้งแต่ -1 (ทิศทางตรงกันข้าม) ไปจนถึง 1 (ทิศทางเดียวกันทุกประการ) โดยมุ่งเน้นไปที่การวางแนวของเวกเตอร์แทนที่จะเป็นขนาดของเวกเตอร์:

$$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

### B. ระยะทางยูคลิด (Euclidean Distance - บรรทัดฐาน $L2$)
ระยะทางยูคลิดจะวัดระยะทางเป็นเส้นตรงระหว่างจุดสองจุดในปริภูมิ $d$ มิติ:

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}$$

---

## 🔬 กรณีศึกษาทางคอมพิวเตอร์วิทัศน์: การติดตามวัตถุหลายชิ้น (DeepSORT) (Multi-Object Tracking)
ในกระบวนการทำงานสำหรับการติดตามวัตถุตามสถาปัตยกรรม YOLO (เช่น DeepSORT หรือ BoT-SORT ที่ใช้ในโหมดการติดตามของ YOLO) เวกเตอร์ฝังตัวมีความสำคัญอย่างยิ่งต่อ **การระบุตัวตนซ้ำ (Re-Identification หรือ Re-ID)**:

1.  **เฟรมที่ 1:** YOLO ตรวจจับวาล์วควบคุม (`control-valve`) และสกัดเวกเตอร์ฝังตัวขนาด 128 มิติ $\mathbf{u}_1$ ออกมา
2.  **เฟรมที่ 2:** วาล์วดังกล่าวถูกบดบังชั่วคราวอยู่หลังท่อส่ง
3.  **เฟรมที่ 3:** YOLO ตรวจจับวาล์วได้อีกครั้ง และสกัดเวกเตอร์ฝังตัว $\mathbf{u}_2$ ออกมา
4.  **การจับคู่เชื่อมโยง (Association):** ตัวติดตาม (Tracker) จะคำนวณหาความคล้ายคลึงของโคไซน์ระหว่าง $\mathbf{u}_2$ กับเส้นทางติดตามที่ยังทำงานอยู่ทั้งหมด เนื่องจากความคล้ายคลึง $\text{Similarity}(\mathbf{u}_1, \mathbf{u}_2) = 0.94$ ระบบจึงยืนยันว่าเป็นวาล์วตัวเดิมและยังคงรักษาหมายเลขติดตาม (Track ID) เดิมไว้ แม้ว่าพิกัดตำแหน่งจะเปลี่ยนแปลงไปก็ตาม

---

## 💻 การคำนวณความคล้ายคลึงด้วย NumPy ในภาษา Python (Python Similarity Calculation - NumPy)

```python
import numpy as np

def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot_product / (norm_u * norm_v)

# Example 128-d embedding vectors
emb_valve1 = np.random.rand(128)
emb_valve2 = emb_valve1 + np.random.normal(0, 0.05, 128)  # Similar vector
emb_flange = np.random.rand(128)  # Unrelated vector

sim_valve = cosine_similarity(emb_valve1, emb_valve2)
sim_flange = cosine_similarity(emb_valve1, emb_flange)

print(f"Similarity (Valve 1 vs Valve 2): {sim_valve:.4f}")  # Should be close to 1.0
print(f"Similarity (Valve 1 vs Flange): {sim_flange:.4f}")
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX39_Fine_Tuning_TH\|EX39: การปรับจูนอย่างละเอียด (Fine-Tuning)]]
*   [[EX41_Bounding_Box_TH\|EX41: พิกัดกรอบล้อมรอบ (Bounding Box Coordinates)]]
*   กลับไปยังแผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
