import jetson_inference
import jetson_utils
import numpy as np
import cv2

# 1. Load the model using segNet
# Note: input_blob and output_blob must match your ONNX names (input_0, output_0)
net = jetson_inference.segNet(argv=[
    '--model=ONNX/int32/final_jetson_model_crop.onnx', 
    '--labels=ONNX/int32/labels.txt', 
    '--colors=ONNX/int32/colors.txt',
    '--input-blob=input_0', 
    '--output-blob=output_0'
])

# 2. Setup Camera and Display
# '/dev/video0' for USB, 'csi://0' for Raspberry Pi Cam
camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4") 
display = jetson_utils.videoOutput("display://0") 

while display.IsStreaming():
    # Capture the image (lives in GPU memory)
    img = camera.Capture()

    if img is None:
        continue

    h = img.height
    w = img.width

    left = (w // 2) - 112
    top = h - 128
    right = (w // 2) + 112
    bottom = h

    patch = jetson_utils.cudaAllocMapped(width=224, height=128, format=img.format)
    jetson_utils.cudaCrop(img, patch, (left, top, right, bottom))

    # Inference & Overlay
    # filter-mode='point' is faster for the Nano
    net.Process(patch)
    net.Mask(patch, width=224, height=128)

    mask_array = jetson_utils.cudaToNumpy(net.GetMask())

    # Class IDs: 1 = Red, 2 = White
    red_indices = np.where(mask_array == 1)[1]
    white_indices = np.where(mask_array == 2)[1]

    if len(red_indices) > 0:
        red_center = np.mean(red_indices)
        error = red_center - 112 # 112 is the center of the 224-width patch
        print(f"🔴 Red Strip Found! Error: {error:.2f}")
    
    # 5. Visual Feedback
    display.Render(patch)
    display.SetStatus(f"segNet FPS: {net.GetNetworkFPS():.1f}")
