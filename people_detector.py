"""
Python code for people detection - optimised in 640p for better performace in edge devices.
"""
from ultralytics import YOLO
import cv2

# Load lightweight YOLOv8 model
model = YOLO("yolov8n.pt")

# Limit to only "person" class (COCO class 0)
TARGET_CLASS_ID = 0

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Set lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: downscale for faster inference
    resized_frame = cv2.resize(frame, (640, 480))

    # Run inference with streaming enabled (faster)
    results = model(resized_frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == TARGET_CLASS_ID:
                conf = float(box.conf[0])
                if conf < 0.4:  # Confidence threshold
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_frame, f"Person {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("People Detection (Optimized)", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
