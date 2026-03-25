import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    boxes = results[0].boxes
    names = model.names

    total_objects = len(boxes)

    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"Total Objects: {total_objects}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    y_offset = 60
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf > 0.5:
            label = f"{names[cls_id]} ({conf:.2f})"

            cv2.putText(annotated_frame, label,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

            y_offset += 30

    cv2.imshow("Smart Waste Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()