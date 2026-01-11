import jetson_inference
import jetson_utils
import time

# 1. Load your custom model
net = jetson_inference.detectNet(argv=[
    '--model=ONNX/ssd-mobilenet.onnx',
    '--labels=ONNX/labels.txt',
    '--input-blob=input_0',
    '--output-cvg=scores',
    '--output-bbox=boxes',
    '--threshold=0.3'
])

# 2. Change input to a video file path
# You can also add an output path here: videoOutput("output.mp4")
input_file = jetson_utils.videoSource("../BIP_videos_roboter_cam/small_corr_w_obs_1.mp4") 
display = jetson_utils.videoOutput("display://0") 

patch = jetson_utils.cudaAllocMapped(width=224, height=128, format='rgb8')

print("Processing video... Press Ctrl+C to stop.")

while display.IsStreaming():
    # Capture the next frame from the video file
    img = input_file.Capture()
    
    if img is None: # End of video
        break
    
    h = img.height
    w = img.width

    left = (w // 2) - 112
    top = h - 128
    right = (w // 2) + 112
    bottom = h
    jetson_utils.cudaCrop(img, patch, (left, top, right, bottom))
        
    # Inference
    detections = net.Detect(patch)
    
    # Render and display FPS
    display.Render(patch)
    
    # Get current Network FPS (how fast the AI is running)
    fps = net.GetNetworkFPS()
    print(f"Current AI Speed: {fps:.2f} FPS", end='\r')
    
    display.SetStatus("detectNet | {:.1f} FPS".format(fps))

print(f"\nFinished! Average Network FPS: {net.GetNetworkFPS():.2f}")
