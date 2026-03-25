from ultralytics import YOLO

if __name__ == "__main__":
    # Load small model (fastest for CPU)
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="data/YOLO-Waste-Detection-1/data.yaml",   # your yaml
        epochs=30,              # keep low for CPU
        imgsz=416,              # reduce size → faster
        batch=4,                # small batch for laptop
        device="cpu",           # CPU
        workers=2               # reduce load
    )