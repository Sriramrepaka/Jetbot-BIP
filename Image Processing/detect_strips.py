import jetson_inference
import jetson_utils

# Initialize the segNet object
# We point it to your custom ONNX file and define the input/output names used in the export
net = jetson_inference.segNet(argv=[
    f"--model=ONNX/strip_detector_nano.engine",  # Point to the .engine instead of .onnx
    f"--labels=ONNX/classes.txt",
    f"--colors=colors.txt",
    f"--input-blob=input_0",
    f"--output-blob=output_0"
])

# Setup video source (CSI camera or video file)
camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/u_corr.mp4") 
display = jetson_utils.videoOutput("display://0")

while display.IsStreaming():
    # 1. Capture the image
    img_full = camera.Capture()

    crop_roi = (0, 112, 224, 224)

    img_cropped = jetson_utils.cudaAllocMapped(width=224, height=112, format=img_full.format)
    jetson_utils.cudaCrop(img_full, img_cropped, crop_roi)
    
    # 2. Process Segmentation
    # Note: segNet will automatically handle the resizing and normalization
    net.Process(img_cropped)
    
    # 3. Overlay the segmentation mask
    net.Overlay(img_cropped, width=img.width, height=img.height, filter_mode="linear")
    
    # 4. Render the results
    display.Render(img_cropped)
    display.SetStatus(f"segNet | {net.GetNetworkFPS():.1f} FPS")