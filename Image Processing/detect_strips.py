import jetson_inference
import jetson_utils
import numpy as np
import time

# 1. Load your custom model
# --model: path to your ONNX file
# --input_blob: the name you gave in torch.onnx.export (we used 'input')
# --output_blob: the name you gave (we used 'output')
net = jetson_inference.segNet(argv=[
    f"--model=ONNX/strip_detector.onnx",
    "--input_blob=input",
    "--output_blob=output",
    "--batch-size=1"
])

# 2. Setup Camera and Display
camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/u_corr_2.mp4") 
display = jetson_utils.videoOutput("display://0")

print("Running Strip Detection... Press Ctrl+C to stop.")

while True:
    # Capture full image (224x224)
    img = camera.Capture()
    if img is None: continue

    # 3. Bottom Half Crop (Speed Optimization)
    # Define crop region: (left, top, right, bottom)
    crop_roi = (0, 112, 224, 224) 
    img_cropped = jetson_utils.cudaAllocMapped(width=224, height=112, format=img.format)
    jetson_utils.cudaCrop(img, img_cropped, crop_roi)

    # 4. Inference
    # Process() automatically handles the normalization and TensorRT conversion
    net.Process(img_cropped)
    
    # 5. Get the Mask and Overlay
    # This generates a colorized overlay based on your classes
    net.Overlay(img_cropped, filter_mode='linear')

    # 6. Calculate FPS and Display
    fps = net.GetNetworkFPS()
    
    # Render the result
    display.SetStatus(f"Segmentation | Network FPS: {fps:.1f}")
    display.Render(img_cropped)

    if not display.IsStreaming():
        break