from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")  # better model for GPU

    model.train(
        data="data/YOLO-Waste-Detection-1/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,     
        workers=8,
        amp=True
    )