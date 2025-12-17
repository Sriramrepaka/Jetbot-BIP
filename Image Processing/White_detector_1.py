import cv2
import numpy as np

# --- 1. CONFIGURATION ---

VIDEO_FILE_PATH = 'BIP_videos_roboter_cam/big_corr_w_obs_a_video.mp4' 

# T-SHAPE MASK DEFINITION (Kept as a stable geometric filter)
HORIZONTAL_START_FACTOR = 0.50 
VERTICAL_EXCLUSION_WIDTH_FACTOR = 0.30 

# --- 2. ADAPTIVE CIELAB FILTERING PARAMETERS ---

# A/B Range: Soft filter to reject strong colors (like red, blue, or saturated reflections)
# We make it slightly wider than before, as the adaptive threshold handles most noise.
A_MIN = 110 
A_MAX = 150
B_MIN = 110
B_MAX = 150

LOWER_AB = np.array([0, A_MIN, B_MIN]) # L channel is ignored here (set to 0)
UPPER_AB = np.array([255, A_MAX, B_MAX]) # L channel is ignored here (set to 255)

# Adaptive Threshold Parameters
# BLOCK_SIZE: The size of the neighborhood to calculate the local threshold. Must be odd.
ADAPTIVE_BLOCK_SIZE = 41
# C: Constant subtracted from the mean. Used to fine-tune sensitivity.
ADAPTIVE_C = 8 


# --- 3. FILTER PARAMETERS ---
GAUSSIAN_KERNEL_SIZE = (5, 5) 
MORPH_KERNEL_SIZE = (3, 3) 
MIN_CONTOUR_AREA = 1000 


def generate_white_mask(frame):
    """Processes a single frame using Adaptive Thresholding and the T-Shaped Filter."""
    
    height, width, _ = frame.shape
    
    # 1. PRE-PROCESSING
    blurred_frame = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL_SIZE, 0)
    lab_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)
    
    # --- 2. DUAL FILTERING ---
    
    # A. Color Neutrality Mask (Relies on A/B channels)
    # This filters out strongly colored objects, but keeps white/grey
    ab_mask = cv2.inRange(lab_frame, LOWER_AB, UPPER_AB)
    
    # B. Adaptive Lightness Mask (Relies on L channel)
    # Splits the L channel (index 0)
    l_channel = lab_frame[:, :, 0] 
    
    # Adaptive Threshold: Compares each pixel to its neighbors
    l_mask = cv2.adaptiveThreshold(
        l_channel, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Uses a weighted average in the neighborhood
        cv2.THRESH_BINARY_INV,          # Inverted because white strips are HIGH values
        ADAPTIVE_BLOCK_SIZE, 
        ADAPTIVE_C
    )

    # Combine the two masks (must be white AND achromatic)
    final_ciellab_mask = cv2.bitwise_and(l_mask, ab_mask)
    
    # --- 3. Apply the T-SHAPED MASK (Geometric Filter) ---
    t_mask = np.ones((height, width), dtype=np.uint8) * 255 
    
    # A. Horizontal Exclusion 
    horizontal_cut_row = int(height * HORIZONTAL_START_FACTOR)
    t_mask[0:horizontal_cut_row, 0:width] = 0

    # B. Vertical Exclusion 
    exclusion_half_width = int(width * (VERTICAL_EXCLUSION_WIDTH_FACTOR / 2))
    center_x = width // 2
    t_mask[horizontal_cut_row:height, center_x - exclusion_half_width : center_x + exclusion_half_width] = 0

    # Apply the T-Mask to the CIELAB output
    final_mask_geometry = cv2.bitwise_and(final_ciellab_mask, t_mask)
    
    # 4. Morphological Operations
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8) 
    final_mask = cv2.morphologyEx(final_mask_geometry, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    
    return final_mask, t_mask


def filter_and_draw_contours(frame, mask):
    """Draws contours on the original frame."""
    
    contour_frame = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all accepted contours in Green
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_CONTOUR_AREA:
            cv2.drawContours(contour_frame, [contour], -1, (0, 255, 0), 2)
            # Draw bounding box in Red
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
    return contour_frame

# --- Main Video Processing Loop ---
cap = cv2.VideoCapture(VIDEO_FILE_PATH) 

if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_FILE_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

print("Starting video processing with Adaptive Thresholding...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished reading video file.")
        break

    final_mask, t_mask = generate_white_mask(frame)
    contour_display = filter_and_draw_contours(frame, final_mask)
    
    cv2.imshow('1. Original Frame', frame)
    cv2.imshow('2. Final Binary Mask (ADAPTIVE)', final_mask)
    cv2.imshow('3. Contour Filtering Result (Adaptive)', contour_display)
    
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()