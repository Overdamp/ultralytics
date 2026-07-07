# 🧠 EX36: แผนผังคุณลักษณะใน CNN (Feature Maps in CNNs)

แผนผังคุณลักษณะ (Feature Map หรือที่เรียกว่า Activation Map) คือผลลัพธ์จากชั้นคอนโวลูชัน ทำหน้าที่แสดงการกระจายเชิงพื้นที่ของลักษณะทางภาพเฉพาะเจาะจง (เช่น ขอบ มุม พื้นผิว หรือวัตถุที่มีความหมายเชิงความหมาย - Semantic Objects) ที่ตรวจจับได้โดยตัวกรอง (Kernels) ในชั้นนั้นๆ

---

## 1. มิติของแผนผังคุณลักษณะ: $[C, H, W]$
แผนผังคุณลักษณะจะแสดงในรูปของเทนเซอร์ 3 มิติ:
*   **$C$ (Channels - ช่องสัญญาณ):** จำนวนของตัวกรอง/เคอร์เนลที่ประยุกต์ใช้ในชั้นคอนโวลูชัน โดยแต่ละช่องสัญญาณจะสอดคล้องกับลักษณะทางภาพที่เป็นเอกลักษณ์เฉพาะตัว (เช่น ช่องสัญญาณหนึ่งอาจเน้นขอบแนวนอน ในขณะที่อีกช่องสัญญาณหนึ่งอาจเน้นการสะท้อนแสงของโลหะ)
*   **$H \times W$ (Height and Width - ความสูงและความกว้าง):** ความละเอียดเชิงพื้นที่ของแผนผังคุณลักษณะ ซึ่งระบุว่าลักษณะที่ตรวจจับได้นั้นตั้งอยู่ที่ใดในรูปภาพ

---

## 2. ลำดับชั้นของแผนผังคุณลักษณะ (The Feature Map Hierarchy)
เมื่อรูปภาพถูกส่งลึกเข้าไปในโครงข่ายหลัก (Backbone) ของ CNN (เช่น ตัวสกัดคุณลักษณะ CSPDarknet ของ YOLO) มิติขนาดของแผนผังคุณลักษณะจะเกิดการเปลี่ยนแปลง:

```
[ ภาพอินพุต ] ---> [ ชั้นตื้น (Shallow Layers) ] -------------> [ ชั้นลึก (Deep Layers) ]
  (640x640x3)        - H & W ขนาดใหญ่, C ขนาดเล็ก            - H & W ขนาดเล็ก, C ขนาดใหญ่
                     - ความละเอียดเชิงพื้นที่สูง               - ความละเอียดเชิงพื้นที่ต่ำ
                     - ความหมายเชิงนามธรรมต่ำ                 - ความหมายเชิงนามธรรมสูง
                       (Low semantic abstraction)          (High semantic abstraction)
                     - (ตรวจจับเส้นตรง, วงกลม)               - (ตรวจจับวาล์ว, ปั๊ม ทั้งชิ้น)
```

*   **ชั้นตื้น (Shallow Layers):** คงความแม่นยำในการระบุตำแหน่งสูง (Localization Precision) จึงเหมาะสำหรับการทำนายขอบเขตกรอบล้อมรอบ (Bounding Box) ที่แม่นยำ
*   **ชั้นลึก (Deep Layers):** สูญเสียรายละเอียดเชิงพื้นที่ (เนื่องจากการลดขนาดด้วยระยะก้าว - Stride Downsampling) แต่ได้รับบริบทเชิงความหมายสูง (Semantic Context) จึงเหมาะสำหรับการทำนายคลาส (Class Labels)

---

## 🔬 กรณีศึกษาทางคอมพิวเตอร์วิทัศน์: การแสดงภาพการเปิดใช้งานของ YOLO (Visualizing YOLO Activations)
หากคุณสกัดและพล็อตแผนผังคุณลักษณะของโมเดล YOLO ที่ฝึกสอนด้วยชุดข้อมูล [overall-ptt-object-detection.v11i.yolov11](file:///home/luke/ai_training/ultralytics/datasets/coco8/overall-ptt-object-detection.v11i.yolov11):
1.  **ชั้นที่ 1 (Conv):** การเปิดใช้งาน (Activations) จะสว่างขึ้นตามแนวขอบตรงที่คมชัดของท่อส่ง (Pipelines) และขอบด้านนอกของเกจวัดแบบเข็ม (`analog-gauges`)
2.  **ชั้นที่ 15 (SPPF/Neck):** รายละเอียดเชิงพื้นที่หายไป แต่ความร้อนของการเปิดใช้งาน (Activations) จะปรากฏเป็นแผนที่ความร้อน (Heatmaps) ที่มีศูนย์กลางอยู่ที่วัตถุเป้าหมายโดยตรง เช่น จุดสว่างของการเปิดใช้งานที่เข้มข้นตรงส่วนของตัววาล์วควบคุม (`control-valve`)

---

## 💻 การสกัดแผนผังคุณลักษณะด้วย PyTorch ในภาษา Python (Python PyTorch Feature Map Extraction)

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel, 8 output feature map channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv1(x)

# Create model and dummy image (Batch=1, Channels=1, 64x64)
model = SimpleConvNet()
img = torch.randn(1, 1, 64, 64)

# Extract feature maps
with torch.no_grad():
    feature_maps = model(img) # Shape: [1, 8, 64, 64]

print(f"Feature Map Shape: {feature_maps.shape}")

# To plot the 3rd feature map channel:
# map_to_plot = feature_maps[0, 2, :, :].numpy()
# plt.imshow(map_to_plot, cmap='gray')
# plt.savefig('feature_map_3.png')
```

---
*หัวข้อที่เกี่ยวข้อง:*
*   [[EX35_CNN_TH\|EX35: โครงข่ายประสาทคอนโวลูชัน (Convolutional Neural Networks - CNN)]]
*   [[EX37_Pooling_TH\|EX37: ชั้นพูลลิง (Pooling Layers)]]
*   กลับไปยังแผนการเรียนหลัก: [[YOLO_Learning_Plan\|แผนการเรียน YOLO]]
*   บันทึกความก้าวหน้า: [[learning_journal\|บันทึกการเรียนรู้]]
