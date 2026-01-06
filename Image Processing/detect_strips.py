import cv2
import numpy as np
import onnxruntime as ort
import jetson_utils as jetson
import time

# 1. Load ONNX Model with CUDA Provider
# This allows the ONNX model to run on the Nano's GPU
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # Limit to 2GB
        'cudnn_conv_algo_search': 'DEFAULT',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

session = ort.InferenceSession("ONNX/strip_detector.onnx", providers=providers)
input_name = session.get_inputs()[0].name

# 2. Initialize Camera (jetson.utils is faster than cv2.VideoCapture)
# Using 224x224 input
camera = jetson.utils.videoSource("BIP_videos_roboter_cam/u_corr_2.mp4") # Use "/dev/video0" for USB

# Timing for FPS
prev_time = 0

print("Starting Segmentation (ONNX + jetson_utils)...")

while True:
    # Capture image (returns a jetson.utils.cudaImage)
    img = camera.Capture()
    if img is None: continue

    start_time = time.time()

    # 3. Bottom Half Crop & Pre-processing
    # jetson_utils.cudaCrop is much faster than numpy cropping
    # Crop: (left, top, right, bottom)
    crop_roi = (0, 112, 224, 224) 
    img_cropped = jetson.utils.cudaAllocMapped(width=224, height=112, format=img.format)
    jetson.utils.cudaCrop(img, img_cropped, crop_roi)

    # Convert to Numpy for ONNX Runtime (Standard normalization)
    # We use cudaToNumpy which is a zero-copy operation on Jetson
    array = jetson.utils.cudaToNumpy(img_cropped)
    
    # Pre-process: Resize if needed, Normalize, and Transpose (HWC -> CHW)
    img_input = array.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # 4. Inference
    outputs = session.run(None, {input_name: img_input})
    mask = outputs[0][0][0] # Get the 2D prediction map

    # 5. Post-process & Over-approximation
    mask = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 6. FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - start_time)
    
    # Display results
    # Convert back to BGR for OpenCV display
    display_frame = jetson.utils.cudaToNumpy(img)
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
    
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Nano Camera", display_frame)
    cv2.imshow("Strip Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()