# Autonomous Warehouse Lane Follower & Obstacle Avoidance

This repository contains the **Brain Node**, a ROS-based controller for an NVIDIA Jetson Nano-powered JetBot. The system integrates real-time semantic segmentation for lane tracking and dynamic object detection to navigate warehouse environments autonomously.



## 🚀 Overview
The project implements a hybrid AI perception system that treats a single CSI camera as a virtual radar. By processing two deep learning models simultaneously, the robot perceives its environment and calculates smooth steering commands to stay within lanes while proactively avoiding obstructions.

### Key Features
* **Dual-Model Inference:** Runs `segNet` (Lane Segmentation) and `detectNet` (Object Detection) concurrently using the `jetson-inference` library.
* **Virtual Radar Ray-Casting:** Uses a custom 15-ray "radar" scanning a 120° field of view to detect lane boundaries and obstacles.
* **CUDA Optimization:** Utilizes `cudaAllocMapped` (Zero-Copy memory) to maximize efficiency on the Jetson Nano’s 4GB RAM.
* **Smooth Motion Control:** Implements a Low Pass Filter (LPF) and "ray-jump" filtering to handle poor lighting or gaps in floor markings.

---

## 🛠 Technical Architecture

The `brain_node.py` manages the entire pipeline from raw pixels to motor commands.

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Vision AI** | `jetson-inference` | Executes `.onnx` models for Lane Segmentation and Object Detection. |
| **Preprocessing** | `jetson-utils` | Handles hardware-accelerated CUDA cropping to a 224x128 Region of Interest (ROI). |
| **Logic Engine** | `NumPy` / `Math` | Calculates weighted steering angles via a virtual radar lookup table. |
| **Communication** | `ROS (rospy)` | Publishes `geometry_msgs/Twist` to the `/cmd_vel` topic for motor control. |
| **Post-Processing** | `OpenCV` | Handles mask dilation and visual feedback. |



---

## 🧠 Navigation Logic: The "Virtual Radar"

Instead of traditional line-following, this node uses a sophisticated ray-casting method:

1.  **ROI Cropping:** The camera feed is cropped to the bottom 224 x 128 pixels to focus on the floor and reduce background noise.
2.  **Mask Generation:** `segNet` produces a lane mask. Any objects detected by `detectNet` are "burned" into this mask as impassable obstacles.
3.  **Ray-Casting:** 15 virtual rays are projected from the robot's base. Each ray "walks" through the mask until it hits a boundary or obstacle.
4.  **Weighted Voting:** Rays in the center are weighted more heavily. If an obstacle is detected on the right, the weights for the right-side rays are reduced, forcing the robot to steer left.
5.  **Steering Smoothing:** A steering smoothing factor 0.7xlast_steering + 0.3xnew_angle prevents jittery movement.
6.  **Obstacle Avoidance:** The node also uses LiDAR to detect big obstacles and turns accodingly to avoid the object, LiDAR is prioritised over object and lane detections. 



---

## 📊 Results & Performance

### 1. Inference Performance
* **Resolution:** 224 x 128 (ROI).
* **Lane Model (`segNet`):** ~18-22 FPS.
* **Object Model (`detectNet`):** ~15-20 FPS.
* **End-to-End Latency:** < 60ms.

### 2. Navigation Results

* **Object detection**.
![output_obj_det](https://github.com/user-attachments/assets/477efeed-4d03-4629-a01f-df98cb8a6762)



| Scenario | Accuracy | Behavior |
| :--- | :--- | :--- |
| **Straight Tracking** | 85% | Maintains center-lane position with minimal oscillation. |
| **Obstacle Avoidance** | 92% | Smoothly steers around boxes and returns to lane. |
| **Lane Gaps** | 88% | Ray filtering handles small gaps in floor tape without losing heading. |

---

## ⚙️ Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/lane-follower-jetbot.git](https://github.com/your-username/lane-follower-jetbot.git)
    cd lane-follower-jetbot
    ```

2.  **Dependencies:**
    Ensure you have `jetson-inference` and ROS (Melodic or Noetic) installed on your Jetson Nano.

3.  **Model Setup:**
    Ensure your `.onnx` models and label files are placed in the `/models` directory of the `lane_follower` package.

4.  **Run the Node:**
    ```bash
    rosrun lane_follower brain_node.py
    ```

---

## ⚠️ Known Edge Cases
* **Low Lighting:** High ISO noise can occasionally create false boundaries in the segmentation mask.
* **Total Obstruction:** If the path is 100% blocked, the robot is programmed to maintain its last known valid heading at reduced speed.
