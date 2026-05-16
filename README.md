##Human Detection on Resource-Constrained Edge AI (ESP32-CAM)
## Project Overview

This project demonstrates an end-to-end Edge AI pipeline for real-time Human vs No-Human classification deployed on a resource-constrained embedded system (ESP32-CAM OV3660).

A lightweight deep learning model was:

Trained using TensorFlow / Keras
Based on a dataset derived from COCO dataset
Optimized using post-training quantization
Deployed on ESP32-CAM (Arduino IDE) for real-time inference

This project highlights how modern deep learning models can be adapted for ultra-low power embedded devices.

  Key Features
  Binary classification: Human vs No Human
  Real-time inference using ESP32-CAM
  Fully optimized model using int8 quantization
  Runs on edge device without cloud dependency
  Reduced model size for embedded deployment
  Camera stream + live prediction output
  
##System Architecture

COCO Dataset
     ↓
Data Preprocessing (Filtering Human / Non-Human)
     ↓
TensorFlow / Keras CNN Model
     ↓
Model Training & Validation
     ↓
Post Training Quantization (TFLite INT8)
     ↓
Conversion to Embedded Format
     ↓
ESP32-CAM Deployment (Arduino IDE)
     ↓
Real-time Edge Inference


## Model Details
Framework: TensorFlow / Keras
Model Type: Lightweight CNN classifier
Input Size: (e.g., 96×96 or 128×128 RGB images)
Output Classes:
Human
No Human

##Optimization Techniques
  Post-training INT8 quantization
  Model pruning (if used)
  Input resolution reduction
  Architecture simplification for edge deployment
 
##Dataset
Source: COCO Dataset (filtered subset)
Classes extracted:
Person → Human class
Background / other objects → No Human class
Preprocessing steps:
Image resizing
Normalization
Label mapping
Train / validation split
📦 Model Conversion Pipeline
1. Train Model (Keras)
model.fit(train_data, validation_data, epochs=XX)
2. Convert to TensorFlow Lite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
3. Quantization (INT8)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
4. Convert to C Array (for ESP32)
xxd -i model.tflite > model_data.cc
🔌 Hardware Setup
🧩 Components Used
ESP32-CAM (AI Thinker / OV3660 module)
FTDI Programmer (USB to Serial)
External 5V Power Supply (recommended)
📌 ESP32-CAM Pin Configuration
ESP32-CAM Pin	FTDI Pin
U0R (RX)	TX
U0T (TX)	RX
GND	GND
5V	5V
GPIO0	GND (for flashing mode)
💻 Firmware (Arduino IDE)
Required Libraries
TensorFlow Lite for Microcontrollers
ESP32 Camera Driver
Arduino Core for ESP32
Workflow
Initialize camera (OV3660 configuration)
Capture frame
Preprocess image
Run inference using TFLite Micro
Output prediction (Human / No Human)
Example Inference Flow
if (invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
}

float human_score = output[0];
float nohuman_score = output[1];

if (human_score > 0.5) {
    Serial.println("Human Detected");
} else {
    Serial.println("No Human Detected");
}
📈 Results
✔ Real-time detection on ESP32-CAM
✔ Reduced model size suitable for edge deployment
✔ Low latency inference on constrained hardware
✔ Stable performance under lighting variation
⚠️ Challenges Faced
Memory limitation on ESP32 (SRAM constraints)
Model size reduction without losing accuracy
Camera image preprocessing optimization
TFLite Micro integration complexity
Balancing accuracy vs latency tradeoff
🔮 Future Improvements
Multi-class detection (person, animal, object)
Upgrade to object detection (YOLO-tiny style)
Edge TPU / accelerator support
Motion-triggered inference optimization
Power optimization for battery operation
📁 Repository Structure
├── model/
│   ├── trained_model.h5
│   ├── model.tflite
│   └── model_data.cc
│
├── esp32_firmware/
│   ├── camera.ino
│   ├── inference.ino
│   └── model_settings.h
│
├── training/
│   ├── dataset_preprocessing.py
│   ├── train_model.py
│   └── evaluate.py
│
└── README.md
👨‍💻 Author

Aseed Faisal
Engineering Student | Embedded Systems & FPGA Developer
Focused on: Edge AI, VLSI, FPGA, and AI Hardware Acceleration

📜 License

This project is licensed under the MIT License.
