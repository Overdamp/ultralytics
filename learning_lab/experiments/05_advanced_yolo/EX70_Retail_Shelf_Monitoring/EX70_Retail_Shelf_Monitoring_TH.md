# 🧠 EX70: การตรวจติดตามชั้นวางสินค้าในร้านค้าปลีก (Few-Shot Product Matching)

ในสภาพแวดล้อมการค้าปลีกขนาดใหญ่ การตรวจจับและระบุประเภทของสินค้าที่มีจำนวนหน่วยเก็บสินค้า (SKUs) นับพันประเภทและเปลี่ยนแปลงอยู่ตลอดเวลา ถือเป็นความท้าทายที่สำคัญในด้านคอมพิวเตอร์วิทัศน์ (Computer Vision) การฝึกโมเดล YOLO มาตรฐานเพื่อจำแนกประเภทสินค้ามากกว่า 10,000 รายการนั้น จำเป็นต้องใช้ชุดข้อมูลขนาดมหึมาและต้องทำการฝึกโมเดลใหม่ซ้ำๆ ทุกครั้งที่มีการเปิดตัวสินค้าใหม่

เพื่อแก้ปัญหานี้ เราจึงแบ่งปัญหาออกเป็นสถาปัตยกรรมแบบสองขั้นตอน (Two-stage pipeline) ดังนี้:
1. **การตรวจจับแบบไม่ระบุคลาส (Class-Agnostic Detection):** โมเดล YOLO จะถูกฝึกให้ตรวจจับ *วัตถุสินค้าทั่วไป* บนชั้นวาง โดยส่งคืนเฉพาะกรอบล้อมรอบ (Bounding boxes) โดยไม่สนใจว่าสินค้านั้นคือ SKU ใดโดยเฉพาะ
2. **การสกัดเวกเตอร์ลักษณะเฉพาะและการค้นหาเวกเตอร์ (Few-Shot Feature Embedding & Vector Search):** ภาพสินค้าที่ถูกครอบตัด (Crops) จะถูกส่งไปยังตัวสกัดฟีเจอร์ (เช่น Siamese Network หรือโมเดล ResNet ที่ผ่านการฝึกฝนมาล่วงหน้า) เพื่อสร้างเวกเตอร์ลักษณะเฉพาะ (Embedding vectors) ที่มีความหนาแน่นและมีมิติต่ำ จากนั้นจึงนำเวกเตอร์เหล่านี้ไปเปรียบเทียบกับฐานข้อมูลภาพสินค้าอ้างอิงโดยใช้ความคล้ายคลึงของโคไซน์ (Cosine similarity) หรือการค้นหาเวกเตอร์ (Vector search)

---

## 1. แนวคิดหลักและสูตรทางคณิตศาสตร์

### การตรวจจับแบบไม่ระบุคลาสด้วย YOLO (Class-Agnostic YOLO Detection)
ตัวตรวจจับ YOLO แบบไม่ระบุคลาสจะทำนายกรอบล้อมรอบ ($x, y, w, h$) และคะแนนความมั่นใจ (Confidence score) เพียงค่าเดียวที่ระบุถึงการมีอยู่ของ *สินค้าใดๆ* แทนที่จะคำนวณการกระจายความน่าจะเป็น (Probability distribution) ของแต่ละคลาสจากสินค้านับพันรายการ

### การสกัดฟีเจอร์เวกเตอร์ลักษณะเฉพาะผ่านโมเดล Siamese / ResNet
กำหนดให้ภาพสินค้าที่ถูกครอบตัดคือ $x$ โครงข่ายประสาทเทียมแบบเข้ารหัส $f(\cdot)$ จะทำการแปลง $x$ ไปยังปริภูมิเวกเตอร์ที่มีมิติสูง:

$$\mathbf{z} = f(x) \in \mathbb{R}^d$$

โดยที่ $d$ คือมิติของเวกเตอร์ลักษณะเฉพาะ (ปกติคือ 128, 256 หรือ 512 มิติ) โครงข่ายนี้จะถูกฝึกโดยใช้ฟังก์ชันการสูญเสีย เช่น **Triplet Loss** หรือ **Contrastive Loss** เพื่อให้มั่นใจว่ารูปภาพของสินค้าชนิดเดียวกันจะอยู่ใกล้กันในปริภูมิเวกเตอร์ ในขณะที่รูปภาพของสินค้าต่างชนิดกันจะอยู่ห่างจากกัน

### การค้นหาด้วยความคล้ายคลึงของโคไซน์ (Cosine Similarity Search)
ในการระบุตัวตนของเวกเตอร์ลักษณะเฉพาะของสินค้าที่ต้องการค้นหา $\mathbf{z}_q$ เปรียบเทียบกับฐานข้อมูลเวกเตอร์อ้างอิง $\mathbf{z}_i$ เราจะคำนวณ **ความคล้ายคลึงของโคไซน์ (Cosine Similarity)**:

$$\text{Similarity}(\mathbf{z}_q, \mathbf{z}_i) = \frac{\mathbf{z}_q \cdot \mathbf{z}_i}{\|\mathbf{z}_q\| \|\mathbf{z}_i\|} = \frac{\sum_{j=1}^d z_{q,j} z_{i,j}}{\sqrt{\sum_{j=1}^d z_{q,j}^2} \sqrt{\sum_{j=1}^d z_{i,j}^2}}$$

สินค้า SKU อ้างอิงที่มีค่าความคล้ายคลึงของโคไซน์สูงสุด (และสูงกว่าเกณฑ์ที่กำหนดไว้ล่วงหน้า) จะถูกเลือกเป็นผลลัพธ์การจับคู่สินค้า

---

## 💻 การเขียนโค้ดด้วย Python

สคริปต์ต่อไปนี้จำลองกระบวนการทำงานดังกล่าว: รับข้อมูลจำลองการตรวจจับสินค้าของ YOLO, สร้างเวกเตอร์ลักษณะเฉพาะ และจับคู่กับฐานข้อมูลสินค้าอ้างอิง (Gallery)

```python
import numpy as np

# 1. Simulate embedding database (Gallery) of 3 known SKUs
# Each SKU has a 128-dimensional reference embedding
np.random.seed(42)
embedding_dim = 128
gallery_skus = ["Cola_Classic", "Diet_Cola", "Orange_Soda"]

# Create normalized reference embeddings
gallery_embeddings = {}
for sku in gallery_skus:
    vec = np.random.randn(embedding_dim)
    gallery_embeddings[sku] = vec / np.linalg.norm(vec)

# 2. Simulate detecting a product on a shelf using class-agnostic YOLO
# The detector outputs bounding boxes: [x_min, y_min, x_max, y_max]
detected_boxes = [
    [100, 150, 180, 350],  # Box 1
    [200, 150, 280, 350]   # Box 2
]

# Simulate crop embeddings (adding some noise to the reference embeddings)
query_embeddings = []
# Query 1: A slightly noisy "Cola_Classic"
noise_1 = 0.15 * np.random.randn(embedding_dim)
q_vec_1 = gallery_embeddings["Cola_Classic"] + noise_1
query_embeddings.append(q_vec_1 / np.linalg.norm(q_vec_1))

# Query 2: A slightly noisy "Orange_Soda"
noise_2 = 0.20 * np.random.randn(embedding_dim)
q_vec_2 = gallery_embeddings["Orange_Soda"] + noise_2
query_embeddings.append(q_vec_2 / np.linalg.norm(q_vec_2))

# 3. Perform Cosine Similarity Search
similarity_threshold = 0.70

for idx, q_emb in enumerate(query_embeddings):
    print(f"\nAnalyzing detected product crop {idx + 1} at bounding box {detected_boxes[idx]}:")
    
    best_match = None
    best_score = -1.0
    
    # Compare with each SKU in the gallery
    for sku, ref_emb in gallery_embeddings.items():
        cosine_sim = np.dot(q_emb, ref_emb)
        print(f" - Similarity with {sku}: {cosine_sim:.4f}")
        
        if cosine_sim > best_score:
            best_score = cosine_sim
            best_match = sku
            
    # Decision logic
    if best_score >= similarity_threshold:
        print(f"🎯 MATCHED: SKU classified as '{best_match}' (Confidence: {best_score:.4f})")
    else:
        print(f"⚠️ UNKNOWN: No gallery SKU exceeded the threshold (Best: {best_match} at {best_score:.4f})")
```

---

## 💡 เคล็ดลับจากอาจารย์ (Professor Tips)

*   **ทำไมต้องใช้ Hashing และ Vector DB?** ในซูเปอร์มาร์เก็ตจริงที่มีสินค้ากว่า 50,000 SKUs การค้นหาแบบตรงไปตรงมา (Brute-force search) กับภาพอ้างอิงทั้งหมดสำหรับรูปภาพที่ถูกครอบตัดแต่ละรูปนั้นช้าเกินไป นักพัฒนาจึงเลือกใช้ฐานข้อมูลดัชนีเวกเตอร์ (Vector indexing databases) เช่น **FAISS**, **Milvus** หรือ **Qdrant** ซึ่งใช้อัลกอริทึมการค้นหาเพื่อนบ้านใกล้สุดแบบประมาณค่า (Approximate Nearest Neighbor: ANN เช่น HNSW) เพื่อลดเวลาการค้นหาให้อยู่ในระดับต่ำกว่ามิลลิวินาที (Sub-milliseconds)
*   **การปรับแต่งตัวตรวจจับแบบไม่ระบุคลาส (Agnostic Detector Fine-tuning):** ในการฝึกตัวตรวจจับ YOLO แบบไม่ระบุคลาส ให้ยุบป้ายกำกับ (Labels) ทั้งหมดในชุดข้อมูล (เช่น แบรนด์สินค้าต่างๆ) ให้เหลือหมวดหมู่ทั่วไปเพียงหมวดหมู่เดียวชื่อว่า `product` วิธีนี้จะช่วยให้โมเดลโฟกัสไปที่การทำนายตำแหน่งกรอบล้อมรอบ (Bounding box localization) เพียงอย่างเดียวและฝึกฝนได้ง่ายขึ้น

---

*หัวข้อที่เกี่ยวข้อง:*
*   [[EX71_Sports_Analytics_Tracking_TH]]
*   [[YOLO_Learning_Plan|แผนการเรียน YOLO]]
*   [[learning_journal|บันทึกการเรียนรู้]]
