import cv2
import numpy as np
import onnxruntime as ort
import time
from line_detector import detect_and_draw_lines

# 1. Configuration
MODEL_PATH = "C:/Users/srira/Documents/BIP_Repo/Jetbot-BIP/Image Processing/ONNX/ssd-mobilenet.onnx"
VIDEO_PATH = "big_corr_1.mp4"  # Change this to 0 for Webcam
LABELS = ["BACKGROUND", "steelball", "pipe"] 
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

# 2. Load Model
session = ort.InferenceSession(MODEL_PATH)

# 3. Initialize Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

# FPS Calculation Variables
frame_count = 0
start_total_time = time.time()

print("Processing video... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    h, w, _ = frame.shape
    loop_start = time.time()

    _,frame,_ = detect_and_draw_lines(frame)

    # --- Preprocessing ---
    input_img = cv2.resize(frame, (300, 300))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = (input_img - 127.5) / 127.5 
    input_tensor = np.expand_dims(input_img.transpose(2, 0, 1), axis=0)

    # --- Inference ---
    inputs = {session.get_inputs()[0].name: input_tensor}
    scores, boxes = session.run(None, inputs)

    all_boxes = []
    all_confidences = []
    all_class_ids = []

    # --- Process Detections ---
    for i in range(scores.shape[1]):
        class_scores = scores[0, i, 1:] 
        score = np.max(class_scores)
        
        if score > CONF_THRESHOLD:
            class_id = np.argmax(class_scores) + 1 
            box = boxes[0, i]
            xmin, ymin = int(box[0] * w), int(box[1] * h)
            xmax, ymax = int(box[2] * w), int(box[3] * h)

            all_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            all_confidences.append(float(score))
            all_class_ids.append(class_id)

    # --- NMS and Drawing ---
    indices = cv2.dnn.NMSBoxes(all_boxes, all_confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = all_boxes[i]
            label = LABELS[all_class_ids[i]]
            conf = all_confidences[i]

            # NOTE: Fixed case-sensitivity here to match your LABELS list
            color = (0, 255, 0) if label == "steelball" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Display FPS on the Frame ---
    frame_count += 1
    current_fps = 1.0 / (time.time() - loop_start)
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("SSD-Mobilenet Video Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup and Final Stats ---
end_total_time = time.time()
avg_fps = frame_count / (end_total_time - start_total_time)

cap.release()
cv2.destroyAllWindows()

print("-" * 30)
print(f"Total Frames Processed: {frame_count}")
print(f"Average FPS: {avg_fps:.2f}")
print("-" * 30)