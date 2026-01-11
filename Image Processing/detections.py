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
    '--model=ONNX/ssd-mobilenet.onnx',
    '--labels=ONNX/labels.txt',
    '--input-blob=input_0',
    '--output-cvg=scores',
    '--output-bbox=boxes',
    '--threshold=0.3'
])

camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/small_corr_w_obs_1.mp4") 
display = jetson_utils.videoOutput("display://0") 
display1 = jetson_utils.videoOutput("display://0")

patch = jetson_utils.cudaAllocMapped(width=224, height=128, format='rgb8')
class_mask = jetson_utils.cudaAllocMapped(width=224, height=128, format="gray8")
mask_np = np.zeros((128,224),dtype=np.uint8)


ROBOT_X, ROBOT_Y = 112, 127
RADAR_RADIUS = 120
SCAN_ANGLES = np.linspace(-60, 60, 20)
ray_lookup = []

for angle in SCAN_ANGLES:
    rad = math.radians(angle)
    coords = []
    for r in range(10, RADAR_RADIUS, 10): # Step 10 is much faster
        curr_x = int(ROBOT_X + r * math.sin(rad))
        curr_y = int(ROBOT_Y - r * math.cos(rad))
        if 0 <= curr_x < 224 and 0 <= curr_y < 128:
            coords.append((curr_x, curr_y, r))
    ray_lookup.append(coords)


weights = np.exp(-0.5 * (np.linspace(-1, 1, len(SCAN_ANGLES))**2)) # Center priority
i = 0
detections = []


while display.IsStreaming():
    loop_start = time.time()

    img = camera.Capture()
    if img is None: continue

    h = img.height
    w = img.width

    left = int((w // 2) - 112)
    top = int(h - 128)
    right = int((w // 2) + 112)
    bottom = int(h)

    # 2. CROP THE SHARED PATCH (128x224)
    jetson_utils.cudaCrop(img, patch, (left, top, right, bottom))
    
    
    lane_net.Process(patch)
    detections = obj_net.Detect(patch)


    lane_net.Mask(class_mask, 224, 128)
    #mask_np = cv2.dilate(jetson_utils.cudaToNumpy(class_mask), np.ones((5,5), np.uint8))
    mask_np = jetson_utils.cudaToNumpy(class_mask)

    if detections:
        for obj in detections:
            cv2.rectangle(mask_np, (int(obj.Left), int(obj.Top)),(int(obj.Right),int(obj.Bottom)),255,-1)
    
    radar_distances = []
    for ray in ray_lookup:
        hit_dist = RADAR_RADIUS
        for cx, cy, r in ray:
            # Check Lane
            if mask_np[cy, cx] > 0:
                hit_dist = r; break
            # Check Objects (Only if objects exist)
       
        radar_distances.append(hit_dist)

    weighted_distances = np.array(radar_distances) * weights
    
    center_dist = radar_distances[len(SCAN_ANGLES)//2]

    if center_dist > 100:
        target_angle = 0.0
        max_path_dist = center_dist
        print("Front is clear, maintaining forward heading.")
    else:
        # If front is blocked, find the best weighted alternative
        best_index = np.argmax(weighted_distances)
        target_angle = SCAN_ANGLES[best_index]
        max_path_dist = radar_distances[best_index]
        print(f"Front blocked ({center_dist}px). Steering to: {target_angle:.1f}°")


    # --- VISUAL FEEDBACK ---
    lane_net.Mask(patch, 224, 128) # Show lane detections

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
    
    total_fps = 1.0 / (time.time() - loop_start)
    
    print(f"Total FPS: {total_fps:.1f} | Lane FPS: {lane_net.GetNetworkFPS():.1f} | Obj FPS: {obj_net.GetNetworkFPS():.1f}")
    status = f"Total FPS: {total_fps:.1f} | Lane FPS: {lane_net.GetNetworkFPS():.1f} | Obj FPS: {obj_net.GetNetworkFPS():.1f}"
    display.SetStatus(status)
    display1.Render(img)
    display.Render(patch)
    #print(f"Steer toward: {target_angle:.1f}° | Path Depth: {max_path_dist}px")
    
    
