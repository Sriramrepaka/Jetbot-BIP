import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

#=====================================================================================================================
'''
/usr/src/tensorrt/bin/trtexec \
  --onnx=strip_detector.onnx \
  --saveEngine=strip_detector.engine \
  --fp16 \
  --explicitBatch \
  --workspace=512
'''
#=====================================================================================================================

# --- TENSORRT SETUP ---
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

# --- INITIALIZATION ---
engine_path = "strip_detector.engine"
engine = load_engine(engine_path)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Setup Camera (0 is default CSI/USB camera)
cap = cv2.VideoCapture('../BIP_videos_roboter_cam/u_corr_no_obs_shaky.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

# Timing variables
fps_start_time = 0
fps = 0

print("Starting inference on Jetson Nano...")

while True:
    ret, frame = cap.read()
    if not ret: break

    start_time = time.time()

    # 1. Pre-processing (Bottom Half Crop)
    # Your model expects 112x224. We crop the bottom half of the 224x224 frame.
    roi = frame[112:224, 0:224] 
    
    # Normalization & Transpose (HWC to CHW)
    img = roi.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)).ravel() # Flatten for TensorRT buffer

    # 2. Inference
    np.copyto(inputs[0]['host'], img)
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # 3. Post-processing
    mask = outputs[0]['host'].reshape(112, 224)
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Over-approximation (Dilation) as requested
    kernel = np.ones((5,5), np.uint8)
    

    hsv_min = np.array([0,75,185])
    hsv_max = np.array([180,140,250])
    
    hsv_full = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv_full, hsv_min, hsv_max)

    mask = cv2.bitwise_or(mask,mask_hsv)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 4. Calculate and Display FPS
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    
    # Draw FPS and Crop Line on the original frame for visualization
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.line(frame, (0, 112), (224, 112), (255, 0, 0), 1) # Show where crop starts

    # 5. Show Results
    cv2.imshow("Nano Camera Feed", frame)
    cv2.imshow("Detection Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()