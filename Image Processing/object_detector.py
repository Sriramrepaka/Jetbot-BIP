import torch
import pathlib
import platform
import cv2

# Fix for Windows users loading Linux-trained models
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

class YoloDetector:
    def __init__(self, model_path='Models/best.pt', conf=0.8, iou=0.45):
        """Initializes the detector and loads the model into memory."""
        print(f"Loading model: {model_path}...")
        # Load local yolov5 logic via torch.hub
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = conf
        self.model.iou = iou

        self.classes = self.model.names  # Get class names (Pipe, Steelball)

    def detect_and_draw(self, frame, size=224):
        """
        Takes a single frame (numpy array), runs detection, 
        and returns the frame with bounding boxes drawn.
        """
        # 1. Run Inference
        results = self.model(frame, size=size)

        detections = results.xyxy[0].cpu().numpy()

        # 3. Manually draw each box
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # Convert coordinates to integers
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
            
            # Set color (Blue) and thickness
            color = (255, 0, 0) 
            thickness = 2
            
            # Draw ONLY the rectangle
            cv2.rectangle(frame, start_point, end_point, color, thickness)

        return frame

# --- Example Usage (If running this file directly) ---
if __name__ == "__main__":
    # Initialize the detector
    detector = YoloDetector(model_path='Models/best.pt')

    # Load a test image to verify it works
    test_img = cv2.imread('C:/Users/srira/Documents/BIP_Repo/Jetbot-BIP/BIP_videos_roboter_cam/12 (162).jpg') # Replace with a real image path
    
    if test_img is not None:
        result_img = detector.detect_and_draw(test_img)
        cv2.imshow('Detection Test', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not find test_image.jpg to verify.")