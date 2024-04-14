# raspi-vision
This repository serves as a comprehensive hub for all code related to computer vision projects specifically designed for Raspberry Pi.

- [1. General Specifications](#1-general-specifications)
- [2. Projects](#2-projects)
  - [2.1 Real-Time Object Detection with TensorFlow Lite](#21-real-time-object-detection-with-tensorflow-lite)
    - [2.1.1 Requirements](#211-requirements)
    - [2.1.2 Code](#212-code)
- [3. References](#3-references)

## 1. General Specifications
- **Software**
  - **OS:** Debian Bullseye
- **Hardware**
  - **Board:** Raspberry Pi 4 Model B Rev 1.2
  - **Webcam:** logitech C270 HD

## 2. Projects

### 2.1 Real-Time Object Detection with TensorFlow Lite
This repository presents a demonstration of real-time object detection on a Raspberry Pi using TensorFlow Lite. Inspired by **the official TensorFlow examples library [1]** and the video tutorials by **Paul McWhorter [2]**, this project provides a hands-on exploration of object detection capabilities on resource-constrained devices like the Raspberry Pi platform.

#### 2.1.1 Requirements
- Python 3.x
- OpenCV
- TensorFlow Lite
- Model: efficientdet_lite0.tflite

#### 2.1.2 Code
Please click [here](https://github.com/dafmontenegro/raspi-vision/blob/master/RealTimeObjectDetection/raspi_vision_object_detection.py) to go to the code

## 3. References
[1] Tensorflow. (s. f.). examples/lite/examples/object_detection/raspberry_pi at master · tensorflow/examples. GitHub. https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi

[2] Paul McWhorter. (2023, 25 mayo). Raspberry Pi LESSON 63: Object Detection on Raspberry Pi Using Tensorflow Lite [Vídeo]. YouTube. https://www.youtube.com/watch?v=yE7Ve3U5Slw