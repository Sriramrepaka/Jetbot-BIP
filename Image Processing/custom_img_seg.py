import jetson.inference
import jetson.utils
import numpy as np

# 1. Load your model as an imageNet object
# This allows us to use the .Classify() method to trigger the GPU
net = jetson.inference.imageNet(argv=[
    '--model=ONNX/int32/strip_model.onnx', 
    '--input-blob=input_0', 
    '--output-blob=output_0'
])

camera = jetson.utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4")
display = jetson.utils.videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()
    if img is None: continue

    # 2. Trigger Inference
    # Even though it's a segmentation model, Classify() will run it
    net.Classify(img)
    
    # 3. Steal the raw output from the output buffer
    # We use the name 'output_0' specifically
    output_cuda = net.GetOutput(0) 
    mask_np = jetson.utils.cudaToNumpy(output_cuda)
    
    # 4. Process and visualize
    mask_2d = np.squeeze(mask_np) # Remove batch/channel dims
    frame_np = jetson.utils.cudaToNumpy(img)
    
    # Apply your classes manually
    # Class 1 (White Strip), Class 2 (Red Strip)
    is_rgba = (frame_np.shape[2] == 4)
    frame_np[mask_2d == 1] = [255, 255, 255, 255] if is_rgba else [255, 255, 255]
    frame_np[mask_2d == 2] = [255, 0, 0, 255] if is_rgba else [255, 0, 0]

    # 5. Render
    out_img = jetson_utils.cudaFromNumpy(frame_np)
    display.Render(out_img)