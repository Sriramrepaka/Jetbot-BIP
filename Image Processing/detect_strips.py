import os
import sys

# 1. CRITICAL: Fix for "Illegal Instruction" (Core Dumped)
# This must be set before any other imports
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

try:
    import jetson_inference
    import jetson_utils
    import numpy as np
    import time
except ImportError as e:
    print(f"Required library not found: {e}")
    sys.exit(1)

# 2. INITIALIZATION
# Use the fixed model: 1x3x112x224 (Exported without dynamic_axes)
net = jetson_inference.segNet(argv=[
    "--model=strip_detector_fixed.onnx",
    "--input_blob=input",
    "--output_blob=output",
    "--labels=classes.txt",
    "--input-dims=1x3x112x224",
    "--mean=0.0",
    "--std=1.0",
    "--batch-size=1"
])

# Setup Camera and Display
camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4")  # Change to "/dev/video0" for USB
display = jetson_utils.videoOutput("display://0")

# 3. PRE-ALLOCATE BUFFERS
# We allocate these ONCE to prevent "dmabuf_fd mapped entry NOT found" errors
img_cropped = jetson_utils.cudaAllocMapped(width=224, height=112, format="rgb8")
mask_img = jetson_utils.cudaAllocMapped(width=224, height=112, format="gray8")

print("Strip Detection Running. Press Ctrl+C to exit...")

try:
    while True:
        # Capture full camera frame
        img = camera.Capture()
        if img is None: continue

        # 4. BOTTOM HALF CROP (0, 112, 224, 224)
        # Fast GPU-to-GPU crop
        jetson_utils.cudaCrop(img, img_cropped, (0, 112, 224, 224))

        # 5. INFERENCE
        # Processes the 112x224 crop
        net.Process(img_cropped)

        # 6. GENERATE BINARY MASK (White strips, Black background)
        # Using 'point' filter to avoid color bleeding or hues
        net.Mask(mask_img, width=224, height=112, filter_mode='point')

        # 7. DISPLAY & FPS
        display.Render(mask_img)
        display.SetStatus(f"Segmentation Mode: Binary | FPS: {net.GetNetworkFPS():.1f}")

        if not display.IsStreaming():
            break

except KeyboardInterrupt:
    print("\nExiting gracefully...")

finally:
    # Cleanup display and camera
    print("Shutting down...")