# =================================================================
# File: video_processor.py
# Description: Handles video capture and frame-by-frame processing.
# =================================================================

import cv2
import sys
import time
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("../BIP_videos_roboter_cam/Output_0.mp4",fourcc,22,(224,224))

    total_frames = 0
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or error reading frame.")
            break

        # Call the robust detection function on the current frame
        processed_frame = detect_and_draw_lines(frame)
        out.write(processed_frame)
        
        # Display the resulting frame
        #time.sleep(0.04)
        cv2.imshow('Robust Line Detection Output', processed_frame)

        total_frames += 1

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    end_time = time.time()
    total_time = end_time - start_time

    if total_time > 0:
        avg_fps = total_frames / total_time

    print(avg_fps)

    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Set this to 0 for a webcam, or a string for a video file path.
    # Replace 'path/to/your/video.mp4' with your actual video file path.
    VIDEO_SOURCE = '../BIP_videos_roboter_cam/small_corr_1.mp4'
    # VIDEO_SOURCE = 0 # Uncomment this for webcam testing

    print(f"Starting video processing from source: {VIDEO_SOURCE}")
    process_video_stream(source=VIDEO_SOURCE)