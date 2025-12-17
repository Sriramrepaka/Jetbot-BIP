# =================================================================
# File: video_processor.py
# Description: Handles video capture and frame-by-frame processing.
# =================================================================

import cv2
import sys
# Import the function from the line_detector.py file (assuming both files are in the same directory)
from line_detector import detect_and_draw_lines 

def process_video_stream(source):
    """
    Reads from a video source (camera or file) and processes each frame.

    :param source: 0 for webcam, or a string for a video file path.
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or error reading frame.")
            break

        # Call the robust detection function on the current frame
        processed_frame = detect_and_draw_lines(frame)
        
        # Display the resulting frame
        cv2.imshow('Robust Line Detection Output', processed_frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Set this to 0 for a webcam, or a string for a video file path.
    # Replace 'path/to/your/video.mp4' with your actual video file path.
    VIDEO_SOURCE = '../BIP_videos_roboter_cam/u_corr.mp4'
    # VIDEO_SOURCE = 0 # Uncomment this for webcam testing

    print(f"Starting video processing from source: {VIDEO_SOURCE}")
    process_video_stream(source=VIDEO_SOURCE)