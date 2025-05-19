"""
Python optimised code for pose estimation using YOLO-pose model using 604p resolution 
for edge devices.
"""

from ultralytics import YOLO
import cv2

# Load lightweight pose model
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (640, 480))
    results = model(resized, stream=True)

    for result in results:
        annotated_frame = result.plot()

    cv2.imshow("Pose Estimation (Optimized)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

