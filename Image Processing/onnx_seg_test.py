import cv2
import numpy as np
import onnxruntime as ort
import time


# 1. Load the model
session = ort.InferenceSession("../Image Processing/ONNX/strip_detector_nano.onnx")
input_name = session.get_inputs()[0].name

# 2. Open Video
cap = cv2.VideoCapture("big_corr_w_sun_w_obs_1.mp4")

fps_start_time = 0
fps = 0

print("Starting inference on Jetson Nano...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    start_time = time.time()

    # Pre-processing: Match the training (Resize to 224x224 then crop bottom half)
    img = cv2.resize(frame, (224, 224))
    roi = img[112:224, 0:224] # Get bottom 112 pixels
    
    # Format for model: CHW and Normalization
    blob = roi.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    # 3. Run Model
    outputs = session.run(None, {input_name: blob})
    mask = outputs[0][0][0] # Get the 2D mask
    
    # 4. Threshold & Display (Over-approximate)
    hsv_min = np.array([0,75,185])
    hsv_max = np.array([180,140,250])
    
    hsv_full = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv_full, hsv_min, hsv_max)

    mask = (mask < 0.5).astype(np.uint8) * 255
    #mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1) # Thicken strips

    final_mask = cv2.bitwise_or(mask,mask_hsv)
    final_mask = cv2.dilate(final_mask, np.ones((3,3), np.uint8), iterations=1)

    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    
    # Draw FPS and Crop Line on the original frame for visualization
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.line(frame, (0, 112), (224, 112), (255, 0, 0), 1) # Show where crop starts
    
    time.sleep(0.04)
    cv2.imshow("Original",frame)
    cv2.imshow("Detection", mask)
    cv2.imshow("Final_Detection_2", final_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()