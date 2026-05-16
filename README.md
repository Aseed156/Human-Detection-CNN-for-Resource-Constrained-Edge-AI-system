## Human Detection on Resource-Constrained Edge AI (ESP32-CAM)
# Project Overview

This project demonstrates an end-to-end Edge AI pipeline for real-time Human vs No-Human classification deployed on a resource-constrained embedded system (ESP32-CAM OV3660).

A lightweight deep learning model was:

Trained using TensorFlow / Keras 
Based on a dataset derived from COCO dataset 
Optimized using post-training quantization 
Deployed on ESP32-CAM (Arduino IDE) for real-time inference 

This project highlights how modern deep learning models can be adapted for ultra-low power embedded devices. 

## Key Features 
  Binary classification: Human vs No Human 
  Real-time inference using ESP32-CAM 
  Fully optimized model using int8 quantization 
  Runs on edge device without cloud dependency 
  Reduced model size for embedded deployment 
  Camera stream + live prediction output 
  
## System Architecture

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


## Repository Structure
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

 ## Authors 
Romaisa Malik (Model Design and Training) 
M. Umer (Model Quantization) 
Aseed Faisal (ESP32 Deployment)  

## License

This project is licensed under the MIT License.
