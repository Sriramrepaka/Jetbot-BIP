import jetson_inference
import jetson_utils
import numpy as np

# 1. Initialize Network
net = jetson_inference.tensorNet(argv=[
    "--model=ONNX/strip_detector_nano.engine",
    "--input-blob=input_0",
    "--output-blob=output_0",
    "--input-scale=0.00392156"  # PC-matched 1/255 scale
])

camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4", argv=["--input-width=224", "--input-height=224"])
display = jetson_utils.videoOutput("display://0")

# Pre-allocate GPU memory for the bottom-half crop
img_cropped = jetson_utils.cudaAllocMapped(width=224, height=112, format="rgb8")

while display.IsStreaming():
    img_full = camera.Capture()
    if img_full is None: continue
    
    # 2. GPU-Accelerated Crop (0, 112 to 224, 224)
    jetson_utils.cudaCrop(img_full, img_cropped, (0, 112, 224, 224))
    
    # 3. MANUAL INFERENCE CYCLE
    # Since your build lacks high-level methods, we use the direct execution path
    # This automatically maps the 'img_cropped' memory to the 'input_0' blob
    net.Execute() 
    
    # 4. GET THE OUTPUT
    # GetOutput(0) returns the raw float32 tensor from your Maxwell GPU
    output_cuda = net.GetOutput(0) 
    
    # 5. POST-PROCESSING (Matching your PC logic)
    # Convert to NumPy for thresholding
    mask_np = jetson_utils.cudaToNumpy(output_cuda)
    
    # Apply your PC threshold (Sigmoid 0.5)
    # Multiply by 255 to create a visible white/black mask
    binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
    
    # 6. VISUALIZATION
    # Convert binary mask back to CUDA for display
    mask_visual = jetson_utils.cudaFromNumpy(binary_mask)
    
    # Render the cropped view and show the network stats
    display.Render(img_cropped)
    display.SetStatus(f"Inference Time: {net.GetNetworkTime():.2f}ms | {net.GetNetworkFPS():.1f} FPS")