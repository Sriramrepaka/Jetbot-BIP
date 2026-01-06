import jetson_inference
import jetson_utils

# 1. Load the optimized engine
# The 'segNet' class is specifically for semantic segmentation
net = jetson_inference.segNet(argv=[
    '--model=ONNX/strip_detector_new.engine', 
    '--labels=ONNX/classes.txt',       # Create this file with 'background' and 'strip'
    '--input-blob=input', 
    '--output-blob=output'
])

# 2. Setup the Camera and Display
# Using jetson_utils for high-performance memory management
camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4")  # Change to "/dev/video0" for USB
display = jetson_utils.videoOutput("display://0")

while display.IsStreaming():
    # Capture frame from camera
    img = camera.Capture()

    # Run inference (Segmentation)
    # This automatically resizes the camera frame to 112x224
    net.Process(img)

    # Apply the mask over the original image
    net.Overlay(img, filter_mode='linear')

    # Render to the monitor
    display.Render(img)

    # Print performance data to the window title
    display.SetStatus(f"Model FPS: {net.GetNetworkFPS():.1f}")