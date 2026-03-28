# ♻️ AI-Based Smart Waste Detection System using YOLOv8

An advanced **Computer Vision-based system** that detects and classifies waste objects in real-time using **YOLOv8**. The system supports **image and webcam-based detection**, helping automate waste segregation and contributing to smart city solutions.

---

## 🚀 Project Overview

Waste management is a critical issue in modern urban environments. Manual segregation of waste is inefficient, time-consuming, and prone to errors.

This project presents an **AI-powered solution** that:

* Detects waste objects using deep learning
* Classifies them into multiple categories
* Works in real-time using a webcam or uploaded images

The system is designed to be **scalable, efficient, and adaptable** for real-world applications like smart bins and automated recycling systems.

---

## 🎯 Key Features

* 🔍 Real-time object detection using YOLOv8
* 📷 Webcam-based live detection
* 🖼️ Image upload detection
* 📊 Object counting and detection summary
* 📈 Confidence-based filtering
* 💾 Download detection results
* 🎨 Clean and interactive Streamlit UI

---

## 🧠 Technologies Used

* **Python**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **Streamlit**
* **NumPy & Pandas**

---

## 📊 Dataset Details

* **Source:** Kaggle Garbage Detection Dataset

* **Total Images:** 55,000+

* **Used Subset:**

  * Train: 5000
  * Validation: 1000
  * Test: 500

* **Classes:** 43 categories including:

  * Plastic bottle
  * Paper
  * Aluminum can
  * Glass bottle
  * Organic waste
  * And more...

* **Format:** YOLO format (images + bounding box labels)

---

## 🧠 Model Details

* **Model Used:** YOLOv8n (Nano version)
* **Why YOLOv8?**

  * Fast and efficient
  * Real-time detection capability
  * High accuracy for object detection tasks

### Training Configuration:

* Epochs: 30
* Image Size: 416
* Batch Size: 4 (CPU optimized)
* Device: CPU (initial training)

---

## 🏗️ Project Structure

```
smart-waste-ai/
│
├── data/                         # Dataset
│   └── YOLO-Waste-Detection-1/
│       └── small/
│
├── models/                       # Trained model
│   └── best.pt
│
├── src/
│   ├── train.py                  # Model training
│   ├── webcam.py                # Real-time detection (OpenCV)
│   └── utils/
│
├── app/
│   └── app.py                   # Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone <your-repo-link>
cd smart-waste-ai
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Training the Model

```bash
python src/train.py
```

After training, the model will be saved at:

```
runs/detect/train/weights/best.pt
```

Move it to:

```
models/best.pt
```

---

## 🎥 Real-Time Webcam Detection

```bash
python src/webcam.py
```

* Press **'q' or ESC** to exit
* Detects objects live with bounding boxes and labels

---

## 🌐 Run Streamlit App

```bash
streamlit run app/app.py
```

### Features:

* Upload image for detection
* Start webcam detection
* View object counts
* Download detection results

---

## 📈 Results

* Successfully detects multiple waste objects
* Works on both images and real-time video
* Displays:

  * Bounding boxes
  * Class labels
  * Confidence scores

---

## ⚠️ Limitations

* Performance depends on hardware (CPU vs GPU)
* Confusion between visually similar classes
* Requires proper lighting for accurate detection
* Large number of classes may reduce accuracy

---

## 🔮 Future Scope

* 🚀 Train model on GPU for better accuracy
* 📱 Mobile app integration
* 🗑️ Smart bin automation using IoT
* 📊 Advanced analytics dashboard
* 🔍 Reduce classes for better performance
* ⚡ Deploy on edge devices (Raspberry Pi)

---

## 🎯 Conclusion

This project demonstrates the use of **deep learning and computer vision** to solve real-world problems in waste management. The system provides a scalable solution for smart cities and can be further enhanced for industrial deployment.

---

## 👨‍💻 Author

**Aman Kumar Gupta**
B.Tech Final Year Project (BTP)

---

## ⭐ Acknowledgements

* Kaggle Dataset Contributors
* Ultralytics YOLOv8
* Open Source Community

---
