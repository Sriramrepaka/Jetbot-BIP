import jetson.inference
import jetson.utils
import numpy as np

# Load as segNet but with the FAKE 1-class label file
net = jetson.inference.segNet(argv=[
    '--model=ONNX/int32/strip_model.engine', 
    '--labels=ONNX/int32/fake_labels.txt', # Use the 1-line file here
    '--input-blob=input_0', 
    '--output-blob=output_0'
])

camera = jetson.utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4")
display = jetson.utils.videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()
    if img is None: continue

    # This WILL work now because the labels match the 1-channel model
    net.Process(img)
    
    # STEAL THE DATA: Get the raw output tensor
    # Even though segNet thinks there is 1 class, the raw numbers are still 0, 1, 2
    output_cuda = net.GetOutput(0) 
    mask_np = jetson.utils.cudaToNumpy(output_cuda)
    
    # 3. MANUALLY DRAW (Same logic as before)
    mask_2d = np.squeeze(mask_np)
    frame_np = jetson.utils.cudaToNumpy(img)
    
    # Class 1 = White, Class 2 = Red
    mask_white = (mask_2d == 1)
    mask_red = (mask_2d == 2)

    # Apply colors
    if frame_np.shape[2] == 4: # RGBA
        frame_np[mask_white] = [255, 255, 255, 255]
        frame_np[mask_red] = [255, 0, 0, 255]
    else: # RGB
        frame_np[mask_white] = [255, 255, 255]
        frame_np[mask_red] = [255, 0, 0]

    # 4. RENDER
    out_img = jetson_utils.cudaFromNumpy(frame_np)
    display.Render(out_img)
    
    # Check if we are actually seeing the IDs 1 and 2
    print(f"IDs in mask: {np.unique(mask_2d)}")
