import jetson_inference
import jetson_utils
import numpy as np
import math
import cv2
import time

# 1. Load the model using segNet
# Note: input_blob and output_blob must match your ONNX names (input_0, output_0)
net = jetson_inference.segNet(argv=[
    '--model=ONNX/int32/final_jetson_model_crop.onnx', 
    '--labels=ONNX/int32/labels.txt', 
    '--colors=ONNX/int32/colors.txt',
    '--input-blob=input_0', 
    '--output-blob=output_0'
])

# 2. Setup Camera and Display
# '/dev/video0' for USB, 'csi://0' for Raspberry Pi Cam
camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/u_corr_2.mp4") 
display = jetson_utils.videoOutput("display://0") 

grid_w, grid_h = 224, 128
class_mask = jetson_utils.cudaAllocMapped(width=grid_w, height=grid_h, format="gray8")

kernel = np.ones((5, 5), np.uint8)

while display.IsStreaming():
    # Capture the image (lives in GPU memory)
    img = camera.Capture()

    if img is None:
        continue

    h = img.height
    w = img.width

    left = (w // 2) - 112
    top = h - 128
    right = (w // 2) + 112
    bottom = h

    patch = jetson_utils.cudaAllocMapped(width=224, height=128, format=img.format)
    jetson_utils.cudaCrop(img, patch, (left, top, right, bottom))

    # Inference & Overlay
    # filter-mode='point' is faster for the Nano
    net.Process(patch)
    net.Mask(class_mask, grid_w, grid_h)

    mask_np = jetson_utils.cudaToNumpy(class_mask)

    mask_np = cv2.dilate(mask_np, kernel, iterations=2)

    y_low = 80
    y_high = 35

    mid_low = None
    mid_high = None

    low_pixels = np.where(mask_np[y_low, :] > 0)[0]
    high_pixels = np.where(mask_np[y_high, :] > 0)[0]

    if len(low_pixels) > 0 and len(high_pixels) > 0:

        mid_low = np.mean(low_pixels)
        mid_high = np.mean(high_pixels)

        angle1 = math.degrees(math.atan2(mid_low - 112, 128 - y_low))
        angle2 = math.degrees(math.atan2(mid_high - mid_low, y_low - y_high))

        final_steering = (angle1 * 0.7) + (angle2 * 0.3)

        print(f"Final steering {final_steering}")

    red_color = (255, 0, 0, 255)
    dot_radius = 5
 
    net.Mask(patch)

    if mid_low is not None:
        # Coordinates must be integers
        cx, cy = int(mid_low), int(y_low)
        jetson_utils.cudaDrawCircle(patch, (cx, cy), dot_radius, red_color)

    # 3. Draw Red Dot on the Top Line Midpoint
    if mid_high is not None:
        cx, cy = int(mid_high), int(y_high)
        jetson_utils.cudaDrawCircle(patch, (cx, cy), dot_radius, red_color)

    

    # 5. Visual Feedback
    time.sleep(0.04)
    display.Render(patch)
    display.SetStatus(f"segNet FPS: {net.GetNetworkFPS():.1f}")
