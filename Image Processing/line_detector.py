# =================================================================
# File: line_detector.py
# Description: Contains the highly robust line detection and contour finding logic.
# =================================================================

import cv2
import numpy as np

def detect_and_draw_lines(img_frame):
    """
    Performs robust line detection using Adaptive Thresholding, custom morphological
    operations, and multiple geometric contour filters.

    :param img_frame: The raw image frame (numpy array) from the video stream.
    :return: The frame with detected contours drawn on it (for visualization).
    """
    if img_frame is None:
        return None

    # 1. Define ROI and Setup Visualization
    height, width, _ = img_frame.shape
    roi_start_y = int(height * 0.50) 
    
    roi = img_frame[roi_start_y:height, 0:width]
    mid_x = width // 2
    contour_vis_roi = roi.copy()
    
    # --- 2. PRE-THRESHOLDING: Bilateral Filter ---
    smoothed_roi = cv2.bilateralFilter(roi, 9, 75, 75)

    # 3. Color Space Conversion (BGR to HSV)
    hsv = cv2.cvtColor(smoothed_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- 4. ROBUST THRESHOLDING: Adaptive Brightness (V) ---
    adaptive_mask = cv2.adaptiveThreshold(
        v, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        41, 
        -5
    )

    # --- 5. POST-THRESHOLDING: Optimized Morphological Operations ---
    
    # Note: Your kernels are non-standard. Using them as defined below:
    kernel_small = np.ones((1, 2), np.uint8) 
    kernel_large = np.ones((2, 3), np.uint8) 

    # 5.a. Opening (Noise Removal): Clean up small isolated noise.
    cleaned_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 5.b. Dilation (Thickening): Ensure thin lines are robust.
    mask_thickened = cv2.dilate(cleaned_mask, kernel_large, iterations=2) 

    # 5.c. Closing (Final Connection): Ensure the strip is contiguous.
    final_binary_output = cv2.morphologyEx(mask_thickened, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    
    # --- 6. CONTOUR FINDING AND FILTERING (Area, Circularity) ---
    
    contours, _ = cv2.findContours(final_binary_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_best = None
    right_best = None
    min_dist_left = float('inf')
    min_dist_right = float('inf')
    # Your working filter constants
    MIN_AREA = 190
    MAX_CIRCULARITY = 0.33 #0.33

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area > MIN_AREA and perimeter > 0:
            # Check 1: Circularity (must be low for a long line)
            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity < MAX_CIRCULARITY:
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate distance to the center line
                    dist_to_center = abs(mid_x - cx)
                    
                    # Selection Logic: Nearest to center on the LEFT
                    if cx < mid_x:
                        if dist_to_center < min_dist_left:
                            min_dist_left = dist_to_center
                            left_best = contour
                    
                    # Selection Logic: Nearest to center on the RIGHT
                    else:
                        if dist_to_center < min_dist_right:
                            min_dist_right = dist_to_center
                            right_best = contour

    # Collect the two best candidates
    selected_strips = [c for c in [left_best, right_best] if c is not None]
    #valid_candidates = sorted(valid_candidates, key=cv2.contourArea, reverse=True)[:2]
    
    offset_error = 0
    path_center_x = mid_x
    # Define a 'slight' error value (e.g., 30 pixels for a 224px wide image)
    # This prevents aggressive over-correction
    SLIGHT_STEER = int(mid_x * 0.15)

    # Case A: Both lines detected (Normal steering)
    if left_best is not None and right_best is not None:
        M_left = cv2.moments(left_best)
        M_right = cv2.moments(right_best)
        cx_l = int(M_left["m10"] / M_left["m00"])
        cx_r = int(M_right["m10"] / M_right["m00"])
        path_center_x = (cx_l + cx_r) // 2
        offset_error = path_center_x - mid_x
        
        #cv2.circle(contour_vis_roi, (path_center_x, 20), 5, (0, 255, 0), -1)

    # Case B: Only LEFT line detected (Slightly Move Right)
    elif left_best is not None:
        offset_error = SLIGHT_STEER
        path_center_x = offset_error + mid_x
        cv2.putText(contour_vis_roi, "SLIGHT RIGHT", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Case C: Only RIGHT line detected (Slightly Move Left)
    elif right_best is not None:
        offset_error = -SLIGHT_STEER
        path_center_x = offset_error + mid_x
        cv2.putText(contour_vis_roi, "SLIGHT LEFT", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    
    # Red circle at path center
    cv2.circle(contour_vis_roi, (path_center_x, roi.shape[0]//2), 5, (0, 0, 255), -1)
    # Blue line representing the error
    cv2.line(contour_vis_roi, (mid_x, roi.shape[0]//2), (path_center_x, roi.shape[0]//2), (255, 0, 0), 2)

    for contour in selected_strips:
            cv2.drawContours(contour_vis_roi, [contour], -1, (0, 255, 0), 2)

    img_frame[roi_start_y:height, 0:width] = contour_vis_roi
    
    return selected_strips, img_frame, offset_error
                

# (End of line_detector.py)