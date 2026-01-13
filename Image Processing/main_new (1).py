#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import jetson_inference
import jetson_utils
from geometry_msgs.msg import Twist
import atexit

# --- CONFIGURATION ---
LANE_MODEL = "./ONNX/final_jetson_model_3.onnx"
LANE_LABELS = "./ONNX/int32/labels.txt"
LANE_COLORS = "./ONNX/int32/colors.txt"

OBJ_MODEL = "./ONNX/ssd-mobilenet.onnx"
OBJ_LABELS = "./ONNX/labels.txt"

# CAMARA
VIDEO_SOURCE = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=224, height=224, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"

SPEED_BASE = 0.20        
KP = 0.004               
MIN_PIXELS = 100         

class JetBotAutonomous:
    def __init__(self):
        rospy.init_node('jetbot_bip_pilot')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        print(f"A carregar Lane Model: {LANE_MODEL}...")
        self.lane_net = jetson_inference.segNet(argv=[
            f'--model={LANE_MODEL}',
            f'--labels={LANE_LABELS}',
            f'--colors={LANE_COLORS}',
            '--input-blob=input_0',
            '--output-blob=output_0'
        ])

        print(f"A carregar Object Model: {OBJ_MODEL}...")
        self.obj_net = jetson_inference.detectNet(argv=[
            f'--model={OBJ_MODEL}',
            f'--labels={OBJ_LABELS}',
            '--input-blob=input_0',
            '--output-cvg=scores',
            '--output-bbox=boxes',
            '--threshold=0.3'
        ])

        self.input_width = 224
        self.input_height = 224
        self.class_mask = jetson_utils.cudaAllocMapped(width=self.input_width, height=self.input_height, format="gray8")

        atexit.register(self.stop_robot)
        print("Sistema Pronto!")

    def get_lane_center_with_offset(self, mask):
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) < MIN_PIXELS:
            return None # Cego

        blob_center_x = int(np.mean(x_coords))
        
        
        OFFSET = 70
        target_x = blob_center_x 
        
        if blob_center_x < 60: 
            target_x = blob_center_x + OFFSET
            
        elif blob_center_x > 164:
            target_x = blob_center_x - OFFSET
            
        return max(0, min(224, target_x))

    def drive(self):
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        print("A iniciar condução...")
        twist = Twist()

        while not rospy.is_shutdown() and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            img_resized = cv2.resize(frame, (self.input_width, self.input_height))
            
            cuda_img = jetson_utils.cudaFromNumpy(img_resized)

            self.lane_net.Process(cuda_img)      
            detections = self.obj_net.Detect(cuda_img)

            self.lane_net.Mask(self.class_mask, self.input_width, self.input_height)
            mask_np = jetson_utils.cudaToNumpy(self.class_mask)
            
            mask_uint8 = mask_np.astype(np.uint8)
            mask = cv2.dilate(mask_uint8, np.ones((5,5), np.uint8), iterations=1)

            obstacle_detected = False
            if detections:
                for obj in detections:
                    if obj.Area > 5000:
                         obstacle_detected = True
                         print(f"Obstáculo detetado! Classe: {obj.ClassID}")

            if obstacle_detected:
                twist.linear.x = 0.0
                twist.angular.z = 0.5
            else:
                lane_x = self.get_lane_center_with_offset(mask)
                image_center = 112 

                if lane_x is not None:
                    error = lane_x - image_center
                    angular_z = -float(error) * KP
                    
                    twist.linear.x = SPEED_BASE
                    twist.angular.z = max(min(angular_z, 1.0), -1.0)
                    
                    cv2.circle(img_resized, (lane_x, 150), 5, (0, 255, 0), -1)
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

            self.cmd_vel_pub.publish(twist)

            mask_vis = (mask * 100).astype(np.uint8)
            cv2.imshow("Camera", img_resized)
            cv2.imshow("Mascara", mask_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.stop_robot()

    def stop_robot(self):
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

if __name__ == '__main__':
    try:
        robot = JetBotAutonomous()
        robot.drive()
    except rospy.ROSInterruptException:
        pass