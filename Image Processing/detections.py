import jetson_inference
import jetson_utils
import numpy as np
import math
import cv2
import time

lane_net = jetson_inference.segNet(argv=[
    '--model=ONNX/int32/final_jetson_model_crop.onnx', 
    '--labels=ONNX/int32/labels.txt', 
    '--colors=ONNX/int32/colors.txt',
    '--input-blob=input_0', 
    '--output-blob=output_0'
])

obj_net = jetson_inference.detectNet(argv=[
    '--model=../ONNX/ssd-mobilenet.onnx',
    '--labels=../ONNX/labels.txt',
    '--input-blob=input_0',
    '--output-cvg=scores',
    '--output-bbox=boxes',
    '--threshold=0.3'
])

camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4") 
display = jetson_utils.videoOutput("display://0") 

PATCH_W, PATCH_H = 224, 128
CANVAS_W, CANVAS_H = 300, 300

canvas = jetson_utils.cudaAllocMapped(width=CANVAS_W, height=CANVAS_H, format='rgb8')

# Radar Config: Origin is bottom-center of the 300x300 canvas
# We place the 224-wide patch at X=38 (to center it: (300-224)/2)
# We place the 128-high patch at Y=172 (at the bottom: 300-128)
OFFSET_X, OFFSET_Y = 38, 172
ROBOT_X, ROBOT_Y = 112, 127
RADAR_RADIUS = 120
SCAN_ANGLES = np.linspace(-60, 60, 20)
weights = np.exp(-0.5 * (np.linspace(-1, 1, len(SCAN_ANGLES))**2)) # Center priority

while display.IsStreaming():
    loop_start = time.time()

    img = camera.Capture()
    if img is None: continue

    h = img.height
    w = img.width

    left = (w // 2) - 112
    top = h - 128
    right = (w // 2) + 112
    bottom = h

    # 2. CROP THE SHARED PATCH (128x224)
    patch = jetson_utils.cudaAllocMapped(width=224, height=128, format=img.format)
    jetson_utils.cudaCrop(img, patch, (left, top, right, bottom))

    jetson_utils.cudaMemset(canvas, 0)
    jetson_utils.cudaOverlay(patch, canvas, OFFSET_X, OFFSET_Y)

    lane_net.Process(patch)
    detections = obj_net.Detect(patch, overlay='box,labels') # DetectNet draws boxes directly on patch

    class_mask = jetson_utils.cudaAllocMapped(width=224, height=128, format="gray8")
    lane_net.Mask(class_mask, 224, 128)
    mask_np = cv2.dilate(jetson_utils.cudaToNumpy(class_mask), np.ones((5,5), np.uint8))
    
    radar_distances = []
    for angle in SCAN_ANGLES:
        rad = math.radians(angle)
        hit_dist = RADAR_RADIUS
        
        for r in range(5, RADAR_RADIUS, 4):
            curr_x = int(ROBOT_X + r * math.sin(rad))
            curr_y = int(ROBOT_Y - r * math.cos(rad))
            
            if 0 <= curr_x < 224 and 0 <= curr_y < 128:
                # --- LANE CHECK ---
                px = curr_x - OFFSET_X
                py = curr_y - OFFSET_Y
                if 0 <= px < PATCH_W and 0 <= py < PATCH_H:
                    if mask_np[py, px] > 0:
                        hit_dist = r; break
                
                # --- OBJECT CHECK (DetectNet Bounding Boxes) ---
                # We check if this radar point falls inside any detected bounding box
                for obj in detections:
                    if obj.Left <= curr_x <= obj.Right and obj.Top <= curr_y <= obj.Bottom:
                        hit_dist = min(hit_dist, r)
                        break
                if hit_dist < RADAR_RADIUS: break
        
        radar_distances.append(hit_dist)

    weighted_distances = np.array(radar_distances) * weights
    center_dist = radar_distances[len(SCAN_ANGLES)//2]

    if center_dist > 90: # Forward is safe
        target_angle = 0.0
    else: # Obstacle or L-Turn ahead, find best opening
        target_angle = SCAN_ANGLES[np.argmax(weighted_distances)]

    lane_net.Mask(patch, 224, 128) # Overlay lane colors
    jetson_utils.cudaOverlay(patch, canvas, OFFSET_X, OFFSET_Y)

    for i, angle in enumerate(SCAN_ANGLES):
        rad = math.radians(angle)
        d = radar_distances[i]
        jetson_utils.cudaDrawLine(canvas, (ROBOT_X, ROBOT_Y), 
                                  (int(ROBOT_X + d*math.sin(rad)), int(ROBOT_Y - d*math.cos(rad))), 
                                  (0, 255, 0, 255), 1)
        
    total_fps = 1.0 / (time.time() - loop_start)

    status = f"Total FPS: {total_fps:.1f} | Lane FPS: {lane_net.GetNetworkFPS():.1f} | Obj FPS: {obj_net.GetNetworkFPS():.1f}"
    display.SetStatus(status)
    display.Render(patch)