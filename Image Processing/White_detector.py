import cv2
import numpy as np

def reduce_carpet_noise(image_path):
    # 1. Load the image and Define ROI
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    roi_start_y = int(height * 0.50) 
    roi = img[roi_start_y:height, 0:width]

    # --- 1. PRE-THRESHOLDING: Gaussian Blur for Texture Smoothing ---
    # Apply blur to the ROI before converting to HSV
    #smoothed_roi = cv2.GaussianBlur(roi, (5, 5), 0) # Use a 5x5 kernel for effective smoothing
    #smoothed_roi = cv2.medianBlur(roi, 3) # Use a 5x5 kernel for effective smoothing

    smoothed_roi = cv2.bilateralFilter(roi, 9, 75, 75)

    # 2. Color Space Conversion (BGR to HSV)
    hsv = cv2.cvtColor(smoothed_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 3. Tighter HSV Thresholding (Using parameters from previous successful attempt)
    # [H, S, V]
    lower_white = np.array([0, 0, 140])    
    upper_white = np.array([180, 80, 255]) 

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # --- 4. POST-THRESHOLDING: Morphological Operations for Noise Removal and Line Thickening ---

    kernel_thickening = np.ones((3, 3), np.uint8)
    mask_d1 = cv2.dilate(mask, kernel_thickening, iterations=2)
    # a. Opening: Effectively removes small noise specs (Erosion followed by Dilation)
    # Use a small kernel (3x3) to only target tiny noise.
    kernel_small = np.ones((1, 1), np.uint8) 
    cleaned_mask = cv2.morphologyEx(mask_d1, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # b. Dilation: Aggressively thicken the remaining strip to ensure continuity
    # Use a larger kernel or more iterations here to make thin/broken strips solid.
    #kernel_thickening = np.ones((3, 3), np.uint8) # Moderate size for curve preservation
    final_binary_output = cv2.morphologyEx(mask_d1, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    S_THRESHOLD = 80 # Adjust this value if the red line is still detected
    ret, s_mask = cv2.threshold(s, S_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    combined_mask = cv2.bitwise_and(final_binary_output, s_mask)
    
    
    # 5. Output Visualization
    
    cv2.imshow("1. Original Image", img)
    cv2.imshow("2. Blurred ROI", smoothed_roi)
    cv2.imshow("3. Threshold Mask (Before Morph)", mask)
    #cv2.imshow("4. Final Binary Output (Noise Removed and Thickened)", mask_d1)
    #cv2.imshow("5. Final Binary Output (Noise Removed and Thickened)", cleaned_mask)
    cv2.imshow("6. Final Binary Output (Noise Removed and Thickened)", final_binary_output)
    cv2.imshow("7. Combined mask",combined_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# --- RUN THE FUNCTION ---
# NOTE: Use the path to your current test image (image_46816c.png or similar)
#image_path = '01.png'
image_path = '2 (33).png' 
reduce_carpet_noise(image_path)