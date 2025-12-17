import cv2
import numpy as np
import math

# ==========================================
# 1. CONFIGURATION (CRUCIAL TUNING AREA)
# ==========================================
VIDEO_PATH = 'BIP_videos_roboter_cam/big_corr_1.mp4'  
ROI_START_Y = 0.50          # Mask out top 50% of the frame
MIN_AREA = 100              # Minimum area for a strip segment

# MULTI-ZONE PIXEL COORDINATES (Based on 640px width)
ZONE_1_END = 200 # Shadow zone boundary
ZONE_2_END = 400 # Transition zone boundary

# HOUGH LINE SETTINGS
MIN_HOUGH_LENGTH = 50       
MAX_HOUGH_GAP = 50          

# YCrCb RED SETTINGS (Most stable color space)
CR_MIN = 150 
CB_MAX = 110 

WHITE_LOW_H = 180
WHITE_LOW_S = 240
WHITE_LOW_V = 100   # We only want pixels brighter than 180
WHITE_HIGH_H = 200  # All hues
WHITE_HIGH_S = 255   # Max Saturation allowed for white
WHITE_HIGH_V = 120
# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_rectangular_mask(frame):
    """Creates the rectangular ROI mask."""
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    start_y = int(height * ROI_START_Y)
    cv2.rectangle(mask, (0, start_y), (width, height), 255, thickness=-1)
    return mask

def enhance_red_strips(frame, roi_mask):
    """Detects Red using stable YCrCb space."""
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    ycrcb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2YCrCb)
    
    # YCrCb Thresholding
    lower_red = np.array([0, CR_MIN, 0])
    upper_red = np.array([255, 255, CB_MAX])
    mask_ycrcb = cv2.inRange(ycrcb, lower_red, upper_red)
    
    # Channel Math Fallback (R vs G)
    G = masked_frame[:,:,1].astype(np.float32)
    R = masked_frame[:,:,2].astype(np.float32)
    mask_math = np.zeros_like(mask_ycrcb)
    mask_math[(R - G) > 15] = 255 
    
    final = cv2.bitwise_or(mask_ycrcb, mask_math)
    
    return cv2.dilate(final, np.ones((5,5), np.uint8), iterations=1)



# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
        
    frame = cv2.resize(frame, (640, 480))
    
    # 1. Create ROI Mask
    roi_mask = get_rectangular_mask(frame)
    
    # 2. Process
    mask_red = enhance_red_strips(frame, roi_mask)
    
    # 3. Visualization
    display = frame.copy()
    
    # Draw ROI Box (Yellow)
    h, w = frame.shape[:2]
    start_y = int(h * ROI_START_Y)
    cv2.rectangle(display, (0, start_y), (w, h), (0, 255, 255), 2)



    # Draw Red Contours (Green)
    contours_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_r:
        if cv2.contourArea(cnt) > MIN_AREA:
            cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)


    cv2.imshow('Final Multi-Zone Dynamic Detection', display)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()