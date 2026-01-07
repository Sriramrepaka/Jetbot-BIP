import jetson_inference
import jetson_utils
import numpy as np

# 1. Load using tensorNet instead of segNet to bypass the "1-class" restriction
net = jetson_inference.tensorNet()
net.LoadNetwork(model="ONNX/int32/strip_model.engine", 
                input_blob="input_0", 
                output_blob="output_0")

camera = jetson_utils.videoSource("../BIP_videos_roboter_cam/big_corr_1.mp4")
display = jetson_utils.videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()
    
    # 2. Run Inference
    # This gives us the raw memory pointer
    net.Process(img)
    
    # 3. Pull the 1-channel mask into NumPy
    # Since your shape is (1,1,H,W), we get a 2D array of 0, 1, and 2
    mask_cuda = net.GetOutput(0) # Index 0 is your output_0
    mask_np = jetson_utils.cudaToNumpy(mask_cuda)
    
    # 4. Create your own visualization
    # We create a colored version of the original image
    img_np = jetson_utils.cudaToNumpy(img)
    
    # Find pixels where class is 1 (White) or 2 (Red)
    img_np[mask_np[0] == 1] = [255, 255, 255] # Paint White
    img_np[mask_np[0] == 2] = [255, 0, 0]     # Paint Red
    
    # 5. Render back to the display
    display.Render(jetson_utils.cudaFromNumpy(img_np))
    display.SetStatus(f"Manual Seg | {net.GetNetworkFPS():.1f} FPS")