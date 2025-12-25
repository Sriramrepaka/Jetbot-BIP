import pathlib
import platform

# Check if we are on Windows and fix the PosixPath error
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import torch
import cv2

# 1. Load your custom model
# Ensure 'best.pt' is in the same folder as this script
model = torch.hub.load('ultralytics/yolov5', 'custom', path='Models/best.pt', force_reload=True)

# 2. Set model parameters
model.conf = 0.4  # Confidence threshold
model.iou = 0.45  # NMS IoU threshold

# 3. Open the video file
video_path = '../BIP_videos_roboter_cam/u_corr.mp4'  # Replace with your video filename
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Processing video... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Inference
    # We pass the frame to the model. We specify size=224 to match your training.
    results = model(frame, size=224)

    # 5. Render results back onto the frame
    # results.render() modifies the 'frame' variable by drawing boxes and labels
    annotated_frame = results.render()[0]

    # 6. Display the frame
    cv2.imshow('YOLOv5 Pipe & Steelball Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()