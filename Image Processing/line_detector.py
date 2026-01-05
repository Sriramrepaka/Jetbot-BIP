# =================================================================
# File: line_detector.py
# Description: Contains the highly robust line detection and contour finding logic.
# =================================================================

import cv2
import numpy as np
import math

def cal_angle(M):

    angle = None
    if M['m00'] != 0:
        # Calculate orientation using second-order central moments
        mu20 = M['mu20'] / M['m00']
        mu02 = M['mu02'] / M['m00']
        mu11 = M['mu11'] / M['m00']
    
        # Formula for orientation angle in radians
        angle_rad = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        angle = math.degrees(angle_rad)

    return angle

def get_vertical_extremes(cnt):
    # cnt[:, :, 1] accesses all Y-coordinates in the contour
    top_idx = cnt[:, :, 1].argmin()
    bot_idx = cnt[:, :, 1].argmax()
    
    # Extract the (x, y) tuples
    top_x, top_y = cnt[top_idx][0]
    bot_x, bot_y = cnt[bot_idx][0]
    
    return int(top_x), int(bot_x)

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

    #mask_thickened = cv2.dilate(adaptive_mask, kernel_large, iterations=1)
    
    # 5.a. Opening (Noise Removal): Clean up small isolated noise.
    cleaned_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # 5.b. Dilation (Thickening): Ensure thin lines are robust.
    mask_thickened = cv2.dilate(cleaned_mask, kernel_large, iterations=1) 

    # 5.c. Closing (Final Connection): Ensure the strip is contiguous.
    final_binary_output = cv2.morphologyEx(mask_thickened, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    
    # --- 6. CONTOUR FINDING AND FILTERING (Area, Circularity) ---
    contours, _ = cv2.findContours(final_binary_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_best = None
    right_best = None
    unassigned = None
    min_dist_left = float('inf')
    min_dist_right = float('inf')
    # Your working filter constants
    MIN_AREA = 190
    MAX_CIRCULARITY = 0.45 #0.33
    r_top_x = 0
    r_bot_x = 0
    l_bot_x = 0
    l_top_x = 0
    visible_l = 0
    visible_r = 0
    angle = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > MIN_AREA and perimeter > 0 :
 
            # Check 1: Circularity (must be low for a long line)
            circularity = 4 * np.pi * area / (perimeter**2)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if circularity < MAX_CIRCULARITY and len(approx) < 10:
                
                M = cv2.moments(contour)

                if M['m00'] != 0:
                    # Calculate orientation using second-order central moments
                    angle = cal_angle(M)

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate distance to the center line
                    dist_to_center = abs(mid_x - cx)
                    
                    if -7 < angle < 7:
                        unassigned = contour

                    # Selection Logic: Nearest to center on the LEFT
                    elif cx < mid_x:
                        if dist_to_center < min_dist_left:
                            min_dist_left = dist_to_center
                            #-ve angle value
                            left_best = contour
                    
                    # Selection Logic: Nearest to center on the RIGHT
                    else:
                        if dist_to_center < min_dist_right:
                            min_dist_right = dist_to_center
                             #+ve angle value
                            right_best = contour

    
    if unassigned is not None:
        unassigned_perimeter = cv2.arcLength(unassigned, True)
        unassigned_area = cv2.contourArea(unassigned)
        cir = 4 * np.pi * unassigned_area / (unassigned_perimeter**2)
        if 250 < unassigned_area < 400 and cir < 0.11:
            if left_best is not None and right_best is None:
                left_best = np.vstack((left_best, unassigned))
                print('Unassigned')
            elif right_best is not None and left_best is None:
                right_best = np.vstack((right_best, unassigned))
                print('Unassigned')
        elif right_best is None and left_best is None:
            M = cv2.moments(unassigned)
            angle = cal_angle(M)
            if angle < 0 :
                left_best = unassigned
            else :
                right_best = unassigned

                

    if right_best is None or left_best is None:
        if left_best is not None:
            M = cv2.moments(left_best)
            angle = cal_angle(M)
            if angle > 0:
                right_best = left_best
                left_best = None
        elif right_best is not None:
            M = cv2.moments(right_best)
            angle = cal_angle(M)
            if angle < 0:
                left_best = right_best
                right_best = None


    # Collect the two best candidates
    cv2.drawContours(contour_vis_roi, [left_best], -1, (0, 0, 255), 2) #red
    cv2.drawContours(contour_vis_roi, [right_best], -1, (255, 255, 255), 2) #white
    

    # Case A: Left case
    if left_best is not None:
        l_top_x, l_bot_x = get_vertical_extremes(left_best)
        visible_l = 1

    # Case B: Right case
    if right_best is not None:
        r_top_x, r_bot_x = get_vertical_extremes(right_best)
        visible_r = 1

    img_frame[roi_start_y:height, 0:width] = contour_vis_roi
    
    
    return l_top_x, l_bot_x, r_top_x, r_bot_x, visible_l, visible_r
                

# (End of line_detector.py)