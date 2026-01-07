import jetson.inference
import jetson.utils
import numpy as np

# Load as 1-class model (matching your current output behavior)
net = jetson.inference.segNet(argv=[
    "--model=ONNX/strip_detector_nano.engine",
    "--input-blob=input_0",
    "--output-blob=output_0",
    "--input-scale=0.00392156"
])

camera = jetson.utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4")
display = jetson.utils.videoOutput("display://0")

# Pre-allocate mask for class IDs
grid_width, grid_height = net.GetGridSize()
class_mask = jetson.utils.cudaAllocMapped(width=grid_width, height=grid_height, format="gray8")

while display.IsStreaming():
    img = camera.Capture()
    if img is None: continue

    # Run inference
    net.Process(img)
    
    # Get the raw Class ID mask (0=Background, 1=Strip if multi-class)
    # For a 1-class model, index 0 is your detection result
    net.Mask(class_mask, width=grid_width, height=grid_height)
    
    # Convert to NumPy for analysis
    mask_np = jetson.utils.cudaToNumpy(class_mask)
    
    # COUNT DETECTIONS:
    # If your model is 1-class, we check values > 0
    num_detected_pixels = np.count_nonzero(mask_np > 0)
    total_pixels = mask_np.size
    percent_coverage = (num_detected_pixels / total_pixels) * 100

    print(f"Detection Status: {percent_coverage:.2f}% of image covered by strips")
    
    display.Render(img)