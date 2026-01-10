import jetson_inference
import jetson_utils
import numpy as np
import math
import cv2
import time

# 1. Setup
net = jetson_inference.segNet(argv=[
    '--model=ONNX/int32/final_jetson_model_crop.onnx', 
    '--labels=ONNX/int32/labels.txt', 
    '--colors=ONNX/int32/colors.txt',
    '--precision=fp32',
    '--input-blob=input_0', 
    '--output-blob=output_0'
])

camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/u_corr.mp4")
display = jetson_utils.videoOutput("display://0")

# Radar Config
ROBOT_X, ROBOT_Y = 112, 127  # Origin in the 128x224 patch
RADAR_RADIUS = 120          # How far ahead the radar "sees"
SCAN_ANGLES = np.linspace(-50, 50, 20)  # 15 rays from -60 to +60 degrees

while display.IsStreaming():
    img = camera.Capture()
    if img is None: continue

    h = img.height
    w = img.width

    left = (w // 2) - 112
    top = h - 128
    right = (w // 2) + 112
    bottom = h

    # Capture 128x224 Patch
    patch = jetson_utils.cudaAllocMapped(width=224, height=128, format=img.format)
    jetson_utils.cudaCrop(img, patch, (left, top, right, bottom))
    
    net.Process(patch)
    
    # Get mask and amplify for better detection
    class_mask = jetson_utils.cudaAllocMapped(width=224, height=128, format="gray8")
    net.Mask(class_mask, 224, 128)
    mask_np = cv2.dilate(jetson_utils.cudaToNumpy(class_mask), np.ones((5,5), np.uint8))

    # --- RADAR LOGIC ---
    radar_distances = []
    
    for angle in SCAN_ANGLES:
        rad = math.radians(angle)
        hit_dist = RADAR_RADIUS  # Default to max range if nothing hit
        
        # Step along the ray
        for r in range(5, RADAR_RADIUS, 4):
            curr_x = int(ROBOT_X + r * math.sin(rad))
            curr_y = int(ROBOT_Y - r * math.cos(rad))
            
            # Boundary & Hit check
            if 0 <= curr_x < 224 and 0 <= curr_y < 128:
                if mask_np[curr_y, curr_x] > 0: # Hit a Red or White line
                    hit_dist = r
                    break
        
        radar_distances.append(hit_dist)

    # --- PATH PLANNING: Find the Safest Opening ---
    # We find the angle that has the furthest clear distance
    best_index = np.argmax(radar_distances)
    target_angle = SCAN_ANGLES[best_index]
    max_path_dist = radar_distances[best_index]

    # --- VISUAL FEEDBACK ---
    net.Mask(patch, 224, 128) # Show lane detections

    # Draw the Radar Rays
    for i, angle in enumerate(SCAN_ANGLES):
        rad = math.radians(angle)
        dist = radar_distances[i]
        end_x = int(ROBOT_X + dist * math.sin(rad))
        end_y = int(ROBOT_Y - dist * math.cos(rad))
        
        # Green ray if clear, Red if hitting a boundary early
        color = (0, 255, 0, 255) if dist > 40 else (255, 0, 0, 255)
        jetson_utils.cudaDrawLine(patch, (ROBOT_X, ROBOT_Y), (end_x, end_y), color, 1)

    # Draw the Steering Decision (Big Blue Arrow)
    t_rad = math.radians(target_angle)
    tx = int(ROBOT_X + max_path_dist * math.sin(t_rad))
    ty = int(ROBOT_Y - max_path_dist * math.cos(t_rad))
    jetson_utils.cudaDrawLine(patch, (ROBOT_X, ROBOT_Y), (tx, ty), (0, 0, 255, 255), 4)

    time.sleep(0.04)
    display.Render(patch)
    print(f"Steer toward: {target_angle:.1f}° | Path Depth: {max_path_dist}px")
