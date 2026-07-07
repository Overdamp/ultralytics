# 🗺️ YOLO Learning Plan & Roadmap

Welcome to your **Ultralytics YOLO** learning plan! This note is designed to help you track your progress, build custom computer vision models, and organize your studies. It connects seamlessly to other notes in your Obsidian Vault.

---

## 📅 Learning Schedule

| Duration | Learning Topic | Goal / Hands-on Task | References & Vault Links | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Day 1** | **YOLO Basics & Setup** | Learn CLI and Python API basics using local tutorial examples. | [[examples/tutorial.ipynb]]<br>[[Journal/learning_journal\|Daily Journal]]<br>[[experiments/04_basic_yolo/EX51_YOLO_CLI/EX51_YOLO_CLI\|EX51: YOLO CLI]]<br>[[experiments/04_basic_yolo/EX52_Model_Configurations/EX52_Model_Configurations\|EX52: Model Configs]] | 🟢 Completed |
| **Day 2-3** | **Dataset Preparation** | Understand YOLO annotation formats, labeling structures, and folder organization. | [[notebooks/YOLO_Training_Manual.ipynb\|Training Manual]]<br>[[experiments/04_basic_yolo/EX61_YOLO_Dataset_Format/EX61_YOLO_Dataset_Format\|EX61: YOLO Dataset Anatomy]] | 🟡 In Progress |
| **Day 4-5** | **Custom Model Training** | Fine-tune a custom YOLO model (26 classes) on your custom PTT dataset. | [[notebooks/YOLO_Training_Manual.ipynb\|Training Manual]]<br>[[experiments/04_basic_yolo/EX53_Training_Settings/EX53_Training_Settings\|EX53: Training Settings]]<br>[[experiments/04_basic_yolo/EX54_Resuming_Training/EX54_Resuming_Training\|EX54: Resuming Training]] | ⚪ Not Started |
| **Day 6-7** | **Evaluation & Metrics** | Analyze train outputs in the `runs/` folder (mAP, Precision, Recall, Confusion Matrix). | [[notebooks/YOLO_Metrics_Explained.ipynb\|Metrics Guide]]<br>[[Concepts/Object_Detection_Tips\|Detection Tips]]<br>[[experiments/04_basic_yolo/EX55_Evaluation_Modes/EX55_Evaluation_Modes\|EX55: Validation Modes]]<br>[[experiments/04_basic_yolo/EX56_Prediction_Sources/EX56_Prediction_Sources\|EX56: Prediction Sources]]<br>[[experiments/04_basic_yolo/EX57_Bounding_Box_Parsing/EX57_Bounding_Box_Parsing\|EX57: BBox Parsing]]<br>[[experiments/04_basic_yolo/EX62_Inference_Visualization/EX62_Inference_Visualization\|EX62: Inference Viz]] | ⚪ Not Started |
| **Day 8-9** | **Hyperparameter Tuning** | Adjust learning rate, optimizers, and augmentations to combat overfitting. | [[notebooks/YOLO_Hyperparameters_Explained.ipynb\|Hyperparameter Guide]]<br>[[Concepts/YOLO_Loss_Functions\|Loss Functions]]<br>[[experiments/04_basic_yolo/EX64_Callbacks_and_Logging/EX64_Callbacks_and_Logging\|EX64: Callbacks & Logging]] | ⚪ Not Started |
| **Day 10** | **Export & Deployment** | Export weights to high-speed formats like ONNX or TensorRT for optimized runtime. | [[experiments/04_basic_yolo/EX58_Model_Export/EX58_Model_Export\|EX58: Model Export]]<br>[[experiments/04_basic_yolo/EX59_Multi_Task_Modes/EX59_Multi_Task_Modes\|EX59: Multi-Task Modes]]<br>[[experiments/04_basic_yolo/EX60_Streaming_Inference/EX60_Streaming_Inference\|EX60: Streaming Inference]]<br>[[experiments/04_basic_yolo/EX63_Multi_Object_Tracking/EX63_Multi_Object_Tracking\|EX63: Multi-Object Tracking]] | ⚪ Not Started |
| **Day 11-12** | **Advanced YOLO Case Studies** | Build cascaded model pipelines for industrial and autonomous applications. | [[experiments/05_advanced_yolo/EX65_Analog_Gauge_Reader/EX65_Analog_Gauge_Reader\|EX65: Analog Gauge]]<br>[[experiments/05_advanced_yolo/EX66_License_Plate_Recognition/EX66_License_Plate_Recognition\|EX66: LPR]]<br>[[experiments/05_advanced_yolo/EX67_PPE_Compliance_Auditing/EX67_PPE_Compliance_Auditing\|EX67: PPE Auditing]]<br>[[experiments/05_advanced_yolo/EX69_Defect_Detection_SAHI/EX69_Defect_Detection_SAHI\|EX69: Defect SAHI]] | ⚪ Not Started |

---

## 🎯 Progress Checklist

- [x] **Setup & Verification:** Configured workspace customization for AI agents and organized directories.
- [x] **YOLO Basics:** Studied basic API workflows and modes in `tutorial.ipynb`.
- [ ] **Dataset Anatomy:** Inspected annotation structure and `data.yaml` inside `overall-ptt-object-detection.v11i.yolov11` (Refer to [[experiments/04_basic_yolo/EX61_YOLO_Dataset_Format/EX61_YOLO_Dataset_Format\|EX61: YOLO Dataset Format]]).
- [ ] **Baseline Training:** Run first baseline training experiment and log results inside [[Journal/learning_journal\|Daily Journal]] (Refer to [[experiments/04_basic_yolo/EX53_Training_Settings/EX53_Training_Settings\|EX53: Training Settings]]).
- [ ] **Metrics Insight:** Confidently explain difference between mAP@0.5 and mAP@0.5:0.95 (Refer to [[experiments/04_basic_yolo/EX55_Evaluation_Modes/EX55_Evaluation_Modes\|EX55: Evaluation Modes]]).
- [ ] **Custom Inference & Viz:** Successfully write custom Python script to run inference, parse classes, and visualize overlays (Refer to [[experiments/04_basic_yolo/EX57_Bounding_Box_Parsing/EX57_Bounding_Box_Parsing\|EX57: BBox Parsing]] & [[experiments/04_basic_yolo/EX62_Inference_Visualization/EX62_Inference_Visualization\|EX62: Inference Viz]]).

---

## 🧠 Vault Index
*   **Overview & Context:** [[PROJECT_SUMMARY|Project Summary & Dataset Overview]] / [[PROJECT_SUMMARY_TH|TH]]
*   **YOLO Python API Master Guide:** [[experiments/04_basic_yolo/YOLO_Python_API_Guide|YOLO Python API Master Guide]] / [[experiments/04_basic_yolo/YOLO_Python_API_Guide_TH|TH]]
*   **Journal Logs:** [[Journal/learning_journal|Daily log of experiments, bugs, and outcomes]]
*   **Core Concepts:**
    *   [[Concepts/YOLO_Loss_Functions|YOLO Loss Functions (CIoU, DFL, BCE Loss Functions Explained)]]
    *   [[Concepts/Object_Detection_Tips\|Tips on Object Detection and Overfitting Prevention]]
*   **YOLO Basics Module (`04_basic_yolo`):**
    *   [[experiments/04_basic_yolo/EX51_YOLO_CLI/EX51_YOLO_CLI\|EX51: YOLO CLI]] / [[experiments/04_basic_yolo/EX51_YOLO_CLI/EX51_YOLO_CLI_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX52_Model_Configurations/EX52_Model_Configurations\|EX52: Model Configs]] / [[experiments/04_basic_yolo/EX52_Model_Configurations/EX52_Model_Configurations_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX53_Training_Settings/EX53_Training_Settings\|EX53: Training Settings]] / [[experiments/04_basic_yolo/EX53_Training_Settings/EX53_Training_Settings_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX54_Resuming_Training/EX54_Resuming_Training\|EX54: Resuming Training]] / [[experiments/04_basic_yolo/EX54_Resuming_Training/EX54_Resuming_Training_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX55_Evaluation_Modes/EX55_Evaluation_Modes\|EX55: Evaluation Modes]] / [[experiments/04_basic_yolo/EX55_Evaluation_Modes/EX55_Evaluation_Modes_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX56_Prediction_Sources/EX56_Prediction_Sources\|EX56: Prediction Sources]] / [[experiments/04_basic_yolo/EX56_Prediction_Sources/EX56_Prediction_Sources_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX57_Bounding_Box_Parsing/EX57_Bounding_Box_Parsing\|EX57: BBox Parsing]] / [[experiments/04_basic_yolo/EX57_Bounding_Box_Parsing/EX57_Bounding_Box_Parsing_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX58_Model_Export/EX58_Model_Export\|EX58: Model Export]] / [[experiments/04_basic_yolo/EX58_Model_Export/EX58_Model_Export_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX59_Multi_Task_Modes/EX59_Multi_Task_Modes\|EX59: Multi-Task Modes]] / [[experiments/04_basic_yolo/EX59_Multi_Task_Modes/EX59_Multi_Task_Modes_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX60_Streaming_Inference/EX60_Streaming_Inference\|EX60: Streaming Inference]] / [[experiments/04_basic_yolo/EX60_Streaming_Inference/EX60_Streaming_Inference_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX61_YOLO_Dataset_Format/EX61_YOLO_Dataset_Format\|EX61: YOLO Dataset Format]] / [[experiments/04_basic_yolo/EX61_YOLO_Dataset_Format/EX61_YOLO_Dataset_Format_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX62_Inference_Visualization/EX62_Inference_Visualization\|EX62: Inference Viz]] / [[experiments/04_basic_yolo/EX62_Inference_Visualization/EX62_Inference_Visualization_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX63_Multi_Object_Tracking/EX63_Multi_Object_Tracking\|EX63: Multi-Object Tracking]] / [[experiments/04_basic_yolo/EX63_Multi_Object_Tracking/EX63_Multi_Object_Tracking_TH\|TH]]
    *   [[experiments/04_basic_yolo/EX64_Callbacks_and_Logging/EX64_Callbacks_and_Logging\|EX64: Callbacks and Logging]] / [[experiments/04_basic_yolo/EX64_Callbacks_and_Logging/EX64_Callbacks_and_Logging_TH\|TH]]
*   **Advanced YOLO Case Studies (`05_advanced_yolo`):**
    *   [[experiments/05_advanced_yolo/EX65_Analog_Gauge_Reader/EX65_Analog_Gauge_Reader\|EX65: Analog Gauge Reader]] / [[experiments/05_advanced_yolo/EX65_Analog_Gauge_Reader/EX65_Analog_Gauge_Reader_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX66_License_Plate_Recognition/EX66_License_Plate_Recognition\|EX66: License Plate Recognition]] / [[experiments/05_advanced_yolo/EX66_License_Plate_Recognition/EX66_License_Plate_Recognition_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX67_PPE_Compliance_Auditing/EX67_PPE_Compliance_Auditing\|EX67: PPE Compliance Auditing]] / [[experiments/05_advanced_yolo/EX67_PPE_Compliance_Auditing/EX67_PPE_Compliance_Auditing_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX68_Crowd_Count_Density_Mapping/EX68_Crowd_Count_Density_Mapping\|EX68: Crowd Density Mapping]] / [[experiments/05_advanced_yolo/EX68_Crowd_Count_Density_Mapping/EX68_Crowd_Count_Density_Mapping_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX69_Defect_Detection_SAHI/EX69_Defect_Detection_SAHI\|EX69: Defect Detection SAHI]] / [[experiments/05_advanced_yolo/EX69_Defect_Detection_SAHI/EX69_Defect_Detection_SAHI_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX70_Retail_Shelf_Monitoring/EX70_Retail_Shelf_Monitoring\|EX70: Retail Shelf Monitoring]] / [[experiments/05_advanced_yolo/EX70_Retail_Shelf_Monitoring/EX70_Retail_Shelf_Monitoring_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX71_Sports_Analytics_Tracking/EX71_Sports_Analytics_Tracking\|EX71: Sports Ball Tracking]] / [[experiments/05_advanced_yolo/EX71_Sports_Analytics_Tracking/EX71_Sports_Analytics_Tracking_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX72_Autonomous_Driving_BEV/EX72_Autonomous_Driving_BEV\|EX72: Autonomous Driving BEV]] / [[experiments/05_advanced_yolo/EX72_Autonomous_Driving_BEV/EX72_Autonomous_Driving_BEV_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX73_Smart_Agriculture_Grading/EX73_Smart_Agriculture_Grading\|EX73: Smart Agriculture Grading]] / [[experiments/05_advanced_yolo/EX73_Smart_Agriculture_Grading/EX73_Smart_Agriculture_Grading_TH\|TH]]
    *   [[experiments/05_advanced_yolo/EX74_Thermal_Fire_Detection/EX74_Thermal_Fire_Detection\|EX74: Thermal Fire Detection]] / [[experiments/05_advanced_yolo/EX74_Thermal_Fire_Detection/EX74_Thermal_Fire_Detection_TH\|TH]]

---
*Tip: Open the Graph View in Obsidian to visually explore how all these concepts and logs link together!*
