import jetson_inference
import jetson_utils

# 1. Load the model using segNet
# Note: input_blob and output_blob must match your ONNX names (input_0, output_0)
net = jetson_inference.segNet(argv=[
    '--model=ONNX/int32/strip_model_3.engine', 
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

    # Inference & Overlay
    # filter-mode='point' is faster for the Nano
    net.Process(img, ignore_class='BACKGROUND')
    net.Overlay(img, filter_mode='point')

    # Render results
    display.Render(img)
    display.SetStatus(f"Segmentation | {net.GetNetworkFPS():.1f} FPS")