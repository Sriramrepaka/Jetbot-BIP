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
    mask_thickened = cv2.dilate(cleaned_mask, kernel_large, iterations=1) 

    # 5.c. Closing (Final Connection): Ensure the strip is contiguous.
    final_binary_output = cv2.morphologyEx(mask_thickened, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    
    # --- 6. CONTOUR FINDING AND FILTERING (Area, Circularity) ---
    
    contours, hierarchy = cv2.findContours(final_binary_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Your working filter constants
    MIN_AREA = 190
    MAX_CIRCULARITY = 0.33

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area > MIN_AREA:
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if h > 0 and w > 0 and perimeter > 0:
                
                # Check 1: Circularity (must be low for a long line)
                circularity = 4 * np.pi * area / (perimeter**2)
                
                if circularity < MAX_CIRCULARITY:
                    
                    # Contour has passed filters! Draw it in Green.
                    cv2.drawContours(contour_vis_roi, [contour], -1, (0, 255, 0), 2)
                    
                    # NOTE: POLY FIT CODE WOULD GO HERE for smooth path

    # Put the processed ROI back into the original frame
    img_frame[roi_start_y:height, 0:width] = contour_vis_roi
    
    return img_frame

# (End of line_detector.py)