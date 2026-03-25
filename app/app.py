import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Waste Detection",
    page_icon="♻️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
mode = st.sidebar.radio("Select Mode", ["Image", "Webcam"])

# ---------------- TITLE ----------------
st.title("♻️ Smart Waste Detection System")
st.markdown("### Real-Time Waste Detection using YOLOv8")

# ---------------- FUNCTION ----------------
def process_detections(results):
    boxes = results[0].boxes
    names = model.names

    class_counts = {}
    records = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]

        if conf > confidence:
            class_counts[label] = class_counts.get(label, 0) + 1
            records.append({"Class": label, "Confidence": round(conf, 2)})

    return class_counts, records

# ---------------- IMAGE MODE ----------------
if mode == "Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", width="stretch")

        results = model(img_array, conf=confidence)
        annotated = results[0].plot()

        with col2:
            st.image(annotated, caption="Detection Result", width="stretch")

        # ---------------- ANALYTICS ----------------
        class_counts, records = process_detections(results)

        st.subheader("📊 Detection Summary")

        if class_counts:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Object Count")
            for label, count in class_counts.items():
                st.write(f"🔹 {label}: {count}")

            st.success(f"Total Objects Detected: {sum(class_counts.values())}")

            # Download result
            st.download_button(
                "📥 Download Result Image",
                data=cv2.imencode('.png', annotated)[1].tobytes(),
                file_name="result.png",
                mime="image/png"
            )
        else:
            st.warning("No objects detected")

# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":
    st.subheader("📷 Live Webcam Detection")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Camera not working")
            break

        results = model(frame, conf=confidence)
        annotated = results[0].plot()

        FRAME_WINDOW.image(annotated, channels="BGR")

    camera.release()