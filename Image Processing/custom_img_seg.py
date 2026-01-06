import jetson_inference
import jetson_utils
import numpy as np

# 1. Load the model as a generic TensorRT network
# We use tensorNet instead of segNet to handle the 1-channel output manually
net = jetson_inference.tensorNet(argv=[
    "--model=ONNX/strip_detector_nano.engine",
    "--input-blob=input_0",
    "--output-blob=output_0",
    "--input-scale=0.00392156"  # Matches your training (1/255.0)
])

# 2. Initialize Camera and Display
# Camera is 224x224, but we will crop to 224x112 for the model
camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4")
display = jetson_utils.videoOutput("display://0")

# Buffer for the cropped image (GPU memory)
img_cropped = jetson_utils.cudaAllocMapped(width=224, height=112, format="rgb8")

while display.IsStreaming():
    # 3. Capture and Crop
    img_full = camera.Capture()
    if img_full is None: continue
    
    # Crop to bottom half (0, 112 to 224, 224)
    jetson_utils.cudaCrop(img_full, img_cropped, (0, 112, 224, 224))
    
    # 4. Inference
    # Forward() returns a cudaImage containing the raw model output (logits/probabilities)
    net.Forward(img_cropped)
    output_cuda = net.GetOutput(0) # Get the first output blob
    
    # 5. Thresholding (Matching your PC logic)
    # Convert CUDA buffer to NumPy for easy thresholding
    mask_np = jetson_utils.cudaToNumpy(output_cuda)
    
    # Apply your 0.5 threshold (adjust if your model outputs raw logits vs sigmoid)
    binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
    
    # 6. Visualization
    # Convert binary mask back to CUDA for rendering
    mask_cuda = jetson_utils.cudaFromNumpy(binary_mask)
    
    # Render the cropped camera view
    display.Render(img_cropped)
    
    # Optional: Render the mask in a second window or overlay
    # display.Render(mask_cuda) 

    display.SetStatus(f"Custom Net | {net.GetNetworkFPS():.1f} FPS")