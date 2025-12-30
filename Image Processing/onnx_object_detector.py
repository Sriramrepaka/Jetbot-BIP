import jetson_inference
import jetson_utils
import time

# 1. Load your custom model
net = jetson_inference.detectNet(argv=[
    '--model=../ONNX/ssd-mobilenet.onnx',
    '--labels=../ONNX/labels.txt',
    '--input-blob=input_0',
    '--output-cvg=scores',
    '--output-bbox=boxes',
    '--threshold=0.3'
])

# 2. Change input to a video file path
# You can also add an output path here: videoOutput("output.mp4")
input_file = jetson_utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4") 
display = jetson_utils.videoOutput("display://0") 

print("Processing video... Press Ctrl+C to stop.")

while display.IsStreaming():
    # Capture the next frame from the video file
    img = input_file.Capture()
    
    if img is None: # End of video
        break
        
    # Inference
    detections = net.Detect(img)
    
    # Render and display FPS
    display.Render(img)
    
    # Get current Network FPS (how fast the AI is running)
    fps = net.GetNetworkFPS()
    print(f"Current AI Speed: {fps:.2f} FPS", end='\r')
    
    display.SetStatus("detectNet | {:.1f} FPS".format(fps))

print(f"\nFinished! Average Network FPS: {net.GetNetworkFPS():.2f}")