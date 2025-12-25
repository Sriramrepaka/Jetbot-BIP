import torch
import torch.nn as nn
import cv2
import numpy as np
import time
from torchvision import transforms
from PIL import Image
from line_detector import detect_and_draw_lines

# Import your PilotNet class from your training code
# (Or redefine it here exactly as it was in Colab)

# --- YOU MUST REDEFINE THE CLASS HERE ---
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # This number must match your Colab training exactly!
            nn.Linear(28224, 100), nn.ReLU(), 
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1) 
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# 1. Setup Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PilotNet().to(device)

# Load the weights you downloaded
model.load_state_dict(torch.load('Models/steering_model.pt', map_location=device))
model.eval()
print("Model loaded successfully!")

# 2. Image Preprocessing (Must match training!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture('../BIP_videos_roboter_cam/big_corr_w_sun_w_obs_1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    _ , _, _ = detect_and_draw_lines(frame)

    # Preprocess frame for the model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 3. Predict Steering
    with torch.no_grad():
        prediction = model(img_tensor).item() # Value between -1 and 1

    # 4. Visualize the Prediction
    h, w, _ = frame.shape
    cv2.putText(frame, f"Steer: {prediction:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw a steering "needle"
    center_x = w // 2
    # Scale prediction to pixels (e.g., max 50px movement)
    target_x = int(center_x + (prediction * 50))
    cv2.line(frame, (center_x, h-20), (target_x, h-60), (0, 0, 255), 3)

    time.sleep(0.04)
    cv2.imshow("Steering Model Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()