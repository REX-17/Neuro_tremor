## TinyML-Based Real-Time Tremor Detection System:
A lightweight embedded AI system for real-time tremor detection using TinyML and multi-modal sensor fusion.

#Overview:
This project presents a TinyML-based healthcare monitoring system designed for real-time detection of neurological tremors using embedded hardware.
The system performs on-device inference using a lightweight machine learning model deployed on the ESP32 microcontroller, eliminating dependency on cloud computing and enabling low-latency monitoring.
The project integrates motion and physiological sensing using the MPU6050 IMU sensor and MAX30102 heart rate sensor to improve reliability and provide continuous real-time monitoring.



#Features:
* Real-time tremor detection
* TinyML-based on-device inference
* Multi-modal sensor fusion
* Portable and low-power embedded system
* Cloud-independent operation
* Real-time sensor monitoring dashboard
* Scalable for Parkinson’s symptom detection



# Hardware Components:
* ESP32 Microcontroller
* MPU6050 IMU Sensor
* MAX30102 Pulse Oximeter / Heart Rate Sensor
* Breadboard and jumper wires



## Software & Tools Used:
* Arduino IDE
* Python
* TensorFlow / TensorFlow Lite
* TinyML
* Serial Monitor / Dashboard Interface


# Working Principle:
The MPU6050 sensor measures acceleration and motion data across three axes to detect tremor-related movement patterns.
The MAX30102 sensor records heart rate data as an additional physiological parameter.
The collected sensor data is preprocessed and passed into a lightweight machine learning model trained for tremor analysis.
The trained model is converted into TensorFlow Lite format and deployed on the ESP32 for real-time inference directly on the embedded device.
The output is visualized through a monitoring dashboard for continuous observation and analysis.


# Results and Inferences:
The system successfully demonstrates real-time tremor monitoring and embedded ML inference with low computational overhead.
The implementation shows that TinyML can be effectively used for portable healthcare applications on resource-constrained devices.



# Applications
* Neurological tremor monitoring
* Wearable healthcare systems
* Remote patient monitoring
* Embedded healthcare AI
* Parkinson’s symptom assistance systems



## Future Scope
* Integration with mobile applications
* Improved model accuracy using larger datasets
* Addition of more physiological sensors
* Real-time cloud synchronization
* Clinical validation and testing



# Project Structure
tinyml-tremor-detection/
│
├── README.md
├── Final_Report.pdf
│
├── firmware/
│   └── esp32_firmware.ino
│
├── model/
│   └── model.tflite
│
├── scripts/
│   ├── training_script.py
│   ├── preprocessing_script.py
│   └── dashboard_script.py
│
├── images/
│   ├── hardware_setup.jpg
│   ├── dashboard_output.png
│   ├── demo_image.jpg
│   └── block_diagram.png
│
└── dataset/
    └── sample_dataset.csv
