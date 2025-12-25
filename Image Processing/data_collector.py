import cv2
import os
import pandas as pd
import numpy as np
from line_detector import detect_and_draw_lines

def generate_verified_dataset(video_path, output_folder="steering_dataset"):
    img_dir = os.path.join(output_folder, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(output_folder, "labels.csv")
    
    # --- CHANGE 1: Find the last frame index to avoid overwriting files ---
    existing_files = os.listdir(img_dir)
    if existing_files:
        # Extracts numbers from filenames like 'frame_000010.jpg'
        indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        frame_idx = max(indices) + 1
    else:
        frame_idx = 0

    cap = cv2.VideoCapture(video_path)
    data_list = []

    print(f"--- CONTINUING COLLECTION FROM FRAME {frame_idx} ---")
    print("Keys: W=0, A=-0.5, D=0.5, J=-0.25, L=0.25, Space=Auto, Q=Save & Quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        _, processed_frame, error = detect_and_draw_lines(frame)
        
        max_error = frame.shape[1] // 2
        auto_steering = np.clip(error / max_error, -1.0, 1.0)

        display_frame = processed_frame.copy()
        cv2.putText(display_frame, f"Auto: {auto_steering:.2f} | Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Verify Steering", display_frame)

        key = cv2.waitKey(0) & 0xFF

        final_steering = auto_steering 

        if key == ord('q'):
            break
        elif key == ord('j'): final_steering = -0.25
        elif key == ord('l'): final_steering = 0.25
        elif key == ord('a'): final_steering = -0.5
        elif key == ord('d'): final_steering = 0.5
        elif key == ord('w'): final_steering = 0.0

        # Save image with unique index
        img_name = f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(img_dir, img_name), frame)
        
        data_list.append({
            "image_path": img_name,
            "steering_angle": final_steering
        })
        frame_idx += 1

    # --- CHANGE 2: Append to CSV instead of overwriting ---
    new_data = pd.DataFrame(data_list)
    if os.path.exists(csv_path):
        # Header=False so we don't repeat 'image_path, steering_angle' inside the file
        new_data.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        new_data.to_csv(csv_path, index=False)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session finished. Total frames in {output_folder}: {frame_idx}")

if __name__ == "__main__":
    # Just run the script for each video one by one
    generate_verified_dataset('../BIP_videos_roboter_cam/u_corr_w_sun_2.mp4', 'Data')