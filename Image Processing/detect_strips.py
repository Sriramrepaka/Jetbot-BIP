import jetson.inference
import jetson.utils
import numpy as np

# 1. Initialize segNet (Model is the same)
net = jetson.inference.segNet(argv=[
    "--model=ONNX/strip_detector_nano.engine",
    "--input-blob=input_0",
    "--output-blob=output_0",
    "--input-scale=0.00392156",
    "--labels=ONNX/classes.txt",
    "--colors=ONNX/colors.txt"
])

# 2. Update Source: Point to your video file instead of "csi://0"
# Replace 'path/to/your_video.mp4' with your actual filename
video_path = "../BIP_videos_roboter_cam/big_corr_1.mp4"
input = jetson.utils.videoSource(video_path) 
display = jetson.utils.videoOutput("display://0")

# Buffer for GPU-accelerated cropping (Remains 224x112)
img_cropped = jetson.utils.cudaAllocMapped(width=224, height=112, format="rgb8")

while display.IsStreaming():
    # 3. Capture a frame from the video
    img_full = input.Capture()
    
    # Check if video has ended
    if img_full is None:
        print("End of video stream")
        break
    
    # 4. GPU-Accelerated Crop (Matches your training)
    # This assumes the video is also roughly 224x224 or scaled similarly
    jetson_utils.cudaCrop(img_full, img_cropped, (0, 112, 224, 224))
    
    # 5. Inference and Overlay
    net.Process(img_cropped)
    net.Overlay(img_cropped, filter_mode="linear", alpha=150)
    
    display.Render(img_cropped)
    display.SetStatus(f"Inference: {net.GetNetworkTime():.2f}ms | FPS: {net.GetNetworkFPS():.1f}")