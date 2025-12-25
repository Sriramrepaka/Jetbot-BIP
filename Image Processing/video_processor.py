# =================================================================
# File: video_processor.py
# Description: Handles video capture, line detection, and object detection.
# =================================================================

import cv2
import sys
import time
# Import the function from the line_detector.py file (assuming both files are in the same directory)
from line_detector import detect_and_draw_lines 
from object_detector import YoloDetector

def process_video_stream(source):
    """
    Reads from a video source (camera or file) and processes each frame.

    :param source: 0 for webcam, or a string for a video file path.
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit()

    # 1. Initialize the YOLO Detector once
    # This keeps the model in memory for efficiency
    detector = YoloDetector(model_path='Models/best.pt', conf=0.4)    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("../BIP_videos_roboter_cam/Output_Combined.mp4",fourcc,22,(224,224))

    total_frames = 0
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or error reading frame.")
            break

         # 1. Define ROI and Setup Visualization
        height, width, _ = frame.shape
        roi_start_y = int(height * 0.50) 
        roi = frame[roi_start_y:height, 0:width]


        # 2. First Pass: Line and Contour Detection
        # This draws the red center path and green strip contours
        contours,_,_ = detect_and_draw_lines(frame)

        for contour in contours:
            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)

        # 3. Second Pass: Object Detection (Pipe & Steelball)
        # This draws the YOLO bounding boxes on the same frame
        # We use size=224 to match your model's training resolution
        processed_frame = detector.detect_and_draw(frame, size=224)

        # 4. Resize for VideoWriter (if your source isn't already 224x224)
        final_output = cv2.resize(processed_frame, (224, 224))
        out.write(final_output)
        
        # Display the resulting frame
        cv2.imshow('Robust Line Detection Output', final_output)

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
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Set this to 0 for a webcam, or a string for a video file path.
    # Replace 'path/to/your/video.mp4' with your actual video file path.
    VIDEO_SOURCE = '../BIP_videos_roboter_cam/u_corr.mp4'
    # VIDEO_SOURCE = 0 # Uncomment this for webcam testing

    print(f"Starting video processing from source: {VIDEO_SOURCE}")
    process_video_stream(source=VIDEO_SOURCE)