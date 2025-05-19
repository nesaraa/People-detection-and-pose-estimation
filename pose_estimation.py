"""
Python code for pose estimation using YOLO-pose model.
"""

from ultralytics import YOLO
import cv2

# Load pretrained pose estimation model
model = YOLO("yolov8n-pose.pt")  # Use yolov8s-pose.pt for better performance

cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose detection
    results = model(frame)

    annotated_frame = results[0].plot()  # Draw keypoints and skeletons
    cv2.imshow("Pose Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
