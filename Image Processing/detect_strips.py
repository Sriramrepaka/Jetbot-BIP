import cv2
import numpy as np
import tensorrt as trt
import jetson.utils  # This replaces PyCUDA for memory management
import time

# --- TENSORRT SETUP ---
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine_path = "ONNX/strip_detector.engine"
engine = load_engine(engine_path)
context = engine.create_execution_context()

# Allocate CUDA memory using jetson.utils
# Model input: 1x3x112x224
input_size = (1, 3, 112, 224)
output_size = (1, 1, 112, 224) # Adjust based on your model output shape

# jetson.utils.cudaAllocMapped is "Zero-Copy" memory
input_mem = jetson.utils.cudaAllocMapped(width=224, height=112, format='rgb32f')
output_mem = jetson.utils.cudaAllocMapped(width=224, height=112, format='gray32f')

# Get pointers for TensorRT
bindings = [int(input_mem.ptr), int(output_mem.ptr)]

cap = cv2.VideoCapture("../BIP_videos_roboter_cam/u_corr_no_obs_shaky.mp4")
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    start_time = time.time()

    # 1. Pre-process (Crop)
    roi = frame[112:224, 0:224]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 2. Copy to CUDA memory
    # jetson.utils allows direct conversion from numpy
    cuda_img = jetson.utils.cudaFromNumpy(rgb)
    jetson.utils.cudaConvertColor(cuda_img, input_mem)

    # 3. Inference
    context.execute_v2(bindings=bindings)

    # 4. Post-process
    # Pull the result back from GPU memory to Numpy
    mask_np = jetson.utils.cudaToNumpy(output_mem)
    mask = (mask_np > 0.5).astype(np.uint8) * 255

    hsv_min = np.array([0,75,185])
    hsv_max = np.array([180,140,250])
    
    hsv_full = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv_full, hsv_min, hsv_max)

    mask = cv2.bitwise_or(mask,mask_hsv)
    
    # Dilation
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8))

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'): break

cap.release()