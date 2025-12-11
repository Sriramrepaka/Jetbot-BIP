import cv2
import numpy as np
import math

# ==========================================
# 1. CONFIGURATION (CRUCIAL TUNING AREA)
# ==========================================
VIDEO_PATH = 'big_corr_1.mp4'  
ROI_START_Y = 0.40          # Mask out top 40% of the frame
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

def enhance_white_multi_zone(frame, roi_mask, red_mask):
    """
    White Detection: Splits the frame into three zones for dynamic thresholding.
    """
    height, width = frame.shape[:2]
    
    # Pre-process: Use Blue Channel (Natural Red Suppression) and Top-Hat
    gray = frame[:,:,0] 
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    
    final_combined_mask = np.zeros_like(gray)
    
    # ---------------------------------------------
    # 1. ZONE 1: DEEP SHADOW (LEFT)
    # ---------------------------------------------
    # Low threshold (15) on Top-Hat for maximum signal in dark areas.
    _, mask1 = cv2.threshold(tophat[:, 0:ZONE_1_END], 15, 255, cv2.THRESH_BINARY)
    final_combined_mask[:, 0:ZONE_1_END] = mask1
    
    # ---------------------------------------------
    # 2. ZONE 2: TRANSITION/CENTER (ADAPTIVE THRESHOLD)
    # ---------------------------------------------
    # Adaptive threshold handles the rapid changes in the center.
    center_region = gray[:, ZONE_1_END:ZONE_2_END]
    center_mask = cv2.adaptiveThreshold(center_region, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
                                        31, -5)
    final_combined_mask[:, ZONE_1_END:ZONE_2_END] = center_mask
    
    # ---------------------------------------------
    # 3. ZONE 3: BRIGHT SUNLIGHT (RIGHT)
    # ---------------------------------------------
    # High threshold (40) on the Top-Hat result: Only the core of the tape registers.
    right_region = tophat[:, ZONE_2_END:width]
    _, mask3 = cv2.threshold(right_region, 40, 255, cv2.THRESH_BINARY)
    final_combined_mask[:, ZONE_2_END:width] = mask3
    
    # ---------------------------------------------
    # 4. FINAL FILTERING AND CLEANUP
    # ---------------------------------------------
    
    # Apply ROI Mask (Deletes windows)
    final_combined_mask = cv2.bitwise_and(final_combined_mask, final_combined_mask, mask=roi_mask)
    
    # Red Subtraction
    red_dilated = cv2.dilate(red_mask, np.ones((7,7), np.uint8), iterations=2)
    final_combined_mask[red_dilated > 0] = 0
    
    # Geometry Filter (Hough Transform) - Rejects noise and stitches segments
    line_mask = np.zeros_like(final_combined_mask)
    lines = cv2.HoughLinesP(final_combined_mask, 1, np.pi/180, threshold=40, 
                            minLineLength=50, maxLineGap=50)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) > 25: 
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=10)
                
    return line_mask

def detect_simple_white(frame, roi_mask):
    """Detects white pixels using a simple HSV range."""
    
    # 1. Apply ROI Mask (to avoid windows/ceiling lights)
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    
    # 2. Convert to HSV
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    
    # 3. Define the White Range
    lower_white = np.array([WHITE_LOW_H, WHITE_LOW_S, WHITE_LOW_V])
    upper_white = np.array([WHITE_HIGH_H, WHITE_HIGH_S, WHITE_HIGH_V])
    
    # 4. Create the Mask
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    return mask

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
    mask_white = detect_simple_white(frame, roi_mask)
    
    # 3. Visualization
    display = frame.copy()
    
    # Draw ROI Box (Yellow)
    h, w = frame.shape[:2]
    start_y = int(h * 0.40)
    cv2.rectangle(display, (0, start_y), (w, h), (0, 255, 255), 2)

    # Draw Red Contours (Green)
    contours_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_r:
        if cv2.contourArea(cnt) > MIN_AREA:
            cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)

    # Draw White Contours (Blue)
    contours_w, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_w:
        cv2.drawContours(display, [cnt], -1, (255, 0, 0), 2)

    cv2.imshow('Final Multi-Zone Dynamic Detection', display)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()