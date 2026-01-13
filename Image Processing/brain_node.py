#!/usr/bin/env python3
import rospy
import rospkg
import jetson_inference
import jetson_utils
import numpy as np
import math
import cv2
from geometry_msgs.msg import Twist


class LaneFollowerNode:
    def __init__(self):
        rospy.init_node('lane_follower_node')

        rp = rospkg.RosPack()
        pkg_path = rp.get_path('lane_follower')

        lane_model_path = f"{pkg_path}/models/final_jetson_model_crop_wdl.onnx"
        obj_model_path = f"{pkg_path}/models/ssd-mobilenet.onnx"
        labels_path = f"{pkg_path}/models/labels.txt"
        colors_path = f"{pkg_path}/models/colors.txt"
        
        # 1. Load your Models (Using the paths we set up)
        self.lane_net = jetson_inference.segNet(argv=[
            f'--model={lane_model_path}',
            f'--labels={labels_path}',
            f'--colors={colors_path}',
            '--input-blob=input_0',
            '--output-blob=output_0',
            '--mean-infer=107.53,107.15,105.06',
            '--std-infer=43.58,43.89,43.32',
            '--filter-mode=point' # Keeps edges sharp for the radar
        ])

        self.obj_net = jetson_inference.detectNet(argv=[
            f'--model={obj_model_path}',
            '--input-blob=input_0',
            '--output-cvg=scores',
            '--output-bbox=boxes',
            '--threshold=0.3'
        ])

        self.patch = jetson_utils.cudaAllocMapped(width=224, height=128, format='rgb8')
        self.class_mask = jetson_utils.cudaAllocMapped(width=224, height=128, format="gray8")
        

        ROBOT_X, ROBOT_Y = 112, 127
        self.RADAR_RADIUS = 120
        self.SCAN_ANGLES = np.linspace(-60, 60, 15)
        self.ray_lookup = []

        for angle in self.SCAN_ANGLES:
            rad = math.radians(angle)
            coords = []
            for r in range(15, self.RADAR_RADIUS, 5): # Step 10 is much faster
                curr_x = int(ROBOT_X + r * math.sin(rad))
                curr_y = int(ROBOT_Y - r * math.cos(rad))
                if 0 <= curr_x < 224 and 0 <= curr_y < 128:
                    coords.append((curr_x, curr_y, r))
            self.ray_lookup.append(coords)

        #self.camera = jetson_utils.videoSource("csi://0") # Or your video file
        self.camera = jetson_utils.videoSource("video path")
        
        # 2. Setup ROS Publisher
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # 3. Movement Parameters
        self.speed = 0.15  # m/s
        self.last_steering = 0.0

        rospy.loginfo("Lane Follower Node Initialized")

    def get_steering_angle(self, mask_np):

        weights = np.exp(-0.5 * (np.linspace(-1, 1, len(self.SCAN_ANGLES))**2))
        radar_distances = []
    
        for ray in self.ray_lookup:
            hit_dist = self.RADAR_RADIUS
            for cx, cy, r in ray:
                # Check Lane
                if mask_np[cy, cx] > 0:
                    hit_dist = r; break
                # Check Objects (Only if objects exist)
       
            radar_distances.append(hit_dist)
   
        weighted_distances = np.array(radar_distances) * weights

        MIDDLE_INDICES = range(len(self.SCAN_ANGLES)//2 - 3, len(self.SCAN_ANGLES)//2 + 3) # 5 Middle wedges
        CLEAR_THRESHOLD = 100

        middle_zone_clear = all(radar_distances[i] > CLEAR_THRESHOLD for i in MIDDLE_INDICES)

        current_weights = weights.copy()

        left_obstruction = np.mean(radar_distances[:5]) < 50
        right_obstruction = np.mean(radar_distances[-5:]) < 50

        if right_obstruction:
            # Scale down weights on the right to force steering left
            current_weights[len(self.SCAN_ANGLES)//2:] *= 0.5 
            #print("Obstacle right: Prioritizing left wedges.")
        elif left_obstruction:
            # Scale down weights on the left to force steering right
            current_weights[:len(self.SCAN_ANGLES)//2] *= 0.5
            #print("Obstacle left: Prioritizing right wedges.")

        # After calculating radar_distances, filter out sudden 'jumps'
        filtered_distances = np.array(radar_distances).copy()
        for i in range(1, len(radar_distances) - 1):
        # If a ray is 50px longer than both its neighbors, it's probably a broken line gap
            if radar_distances[i] > radar_distances[i-1] + 40 and radar_distances[i] > radar_distances[i+1] + 40:
                filtered_distances[i] = (radar_distances[i-1] + radar_distances[i+1]) / 2

        weighted_distances = np.array(filtered_distances) * current_weights

        if middle_zone_clear:
            # If the whole middle area is clear, stay straight
            target_angle = 0.0
            #max_path_dist = np.mean([radar_distances[i] for i in MIDDLE_INDICES])
            #print("Middle Zone is clear, maintaining forward heading.")
        else:
            # If middle is blocked, find the best weighted gap
            best_index = np.argmax(weighted_distances)
            target_angle = self.SCAN_ANGLES[best_index]
            #max_path_dist = radar_distances[best_index]
            #print(f"Pathfinding: Steering to {target_angle:.1f}°")

        angle = (target_angle*1.5)/90

        return angle

    def run(self):
        rate = rospy.Rate(20) # 20Hz for smooth motion
        mask_np = np.zeros((128,224),dtype=np.uint8)
        while not rospy.is_shutdown():
            img = self.camera.Capture()
            if img is None: continue
            
            h = img.height
            w = img.width

            left = int((w // 2) - 112)
            top = int(h - 128)
            right = int((w // 2) + 112)
            bottom = int(h)

            jetson_utils.cudaCrop(img, self.patch, (left, top, right, bottom))

            self.lane_net.Process(self.patch)
            detections = self.obj_net.Detect(self.patch)

            self.lane_net.Mask(self.class_mask, 224, 128)
            mask_np = jetson_utils.cudaToNumpy(self.class_mask)
            #dilate the mask
            mask_np = cv2.dilate(mask_np, np.ones((3,3), np.int8))
            
            if detections:
                for obj in detections:
                    cv2.rectangle(mask_np, (int(obj.Left), int(obj.Top)),(int(obj.Right),int(obj.Bottom)),255,-1)

            # Get angle from your AI
            steering_angle = self.get_steering_angle(mask_np)
            
            # Create the ROS Message
            move_msg = Twist()
            move_msg.linear.x = self.speed
            
            # Simple Smoothing (Low Pass Filter) to prevent jitter
            # We mix 70% of the old angle with 30% of the new one
            smooth_steering = (0.7 * self.last_steering) + (0.3 * steering_angle)
            move_msg.angular.z = smooth_steering
            self.last_steering = smooth_steering
            rospy.loginfo(f"Angle is {smooth_steering}")
            # Send to motors
            self.cmd_pub.publish(move_msg)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = LaneFollowerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass