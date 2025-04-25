#!/usr/bin/env python
import rospy
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import math
from geometry_msgs.msg import Point
import numpy as np
import torch
from threading import Thread

class KalmanFilter:
    def __init__(self, dt, std_acc, std_meas):
        self.dt = dt
        self.x = np.matrix([[0], [0], [0], [0]])  # Initial state

        # Define the state transition matrix, measurement matrix, etc.
        self.F = np.matrix([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.matrix([[0.25 * dt**4, 0, 0.5 * dt**3, 0], [0, 0.25 * dt**4, 0, 0.5 * dt**3], [0.5 * dt**3, 0, dt**2, 0], [0, 0.5 * dt**3, 0, dt**2]]) * std_acc**2
        self.R = np.matrix([[std_meas**2, 0], [0, std_meas**2]])
        self.P = np.eye(self.F.shape[1])

    def predict(self):
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F.T + self.Q
        return self.x[0:2]

    def update(self, z):
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.inv(S)
        y = np.matrix(z).T - self.H * self.x
        self.x = self.x + K * y
        I = np.eye(self.H.shape[1])
        self.P = (I - K * self.H) * self.P
        return self.x[0:2]

class CameraObject:
    def __init__(self):
        rospy.init_node('human_detection', anonymous=False)
        self.bridge = CvBridge()
        
        # Load YOLO model and ensure it uses GPU if available
        self.model = YOLO('/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/runs/detect/train8/weights/best.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)  # Use half precision (FP16) if on GPU

        # Image processing parameters
        self.image_center_X = (416 * 0.0002645833) / 2
        self.image_center_Y = (320 * 0.0002645833) / 2
        self.angles = Point()
        self.distance = None
        self.human_center_x = None
        self.human_center_y = None
        self.angle_pub = rospy.Publisher('angle_topic', Point, queue_size=10)
        
        # Initialize Kalman filter
        self.kalman_filter = KalmanFilter(dt=1.0, std_acc=1.0, std_meas=0.5)
        
        # Start threads for each subscriber to handle callbacks concurrently
        self.image_thread = Thread(target=self.image_subscriber)
        self.depth_thread = Thread(target=self.depth_subscriber)
        self.image_thread.start()
        self.depth_thread.start()

    def image_subscriber(self):
        rospy.Subscriber('/left_camera_1/color/image_raw', Image, self.imageCallback)

    def depth_subscriber(self):
        rospy.Subscriber('/left_camera_1/depth/image_raw', Image, self.depthCallback)

    def imageCallback(self, msg):
        cvImage = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        print('start')
        # Ensure the image is resized before passing to YOLO for faster inference
        results = self.model(cvImage)  # Run inference on the GPU if available
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            measured_center_x = (x1 + x2) // 2
            measured_center_y = (y1 + y2) // 2

            # Kalman filter smoothing
            filtered_center = self.kalman_filter.update([measured_center_x, measured_center_y])
            self.human_center_x, self.human_center_y = int(filtered_center[0, 0]), int(filtered_center[1, 0])
        
        cv2.imshow('human_detection', annotated_frame)
        cv2.waitKey(1)

    def depthCallback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if self.human_center_x is not None and self.human_center_y is not None:
            self.distance = depth_image[self.human_center_x, self.human_center_y]
            rospy.loginfo(f'Distance: {self.distance}')

    def angleCalculation(self):
        if self.distance is not None:
            x_difference = round(((self.human_center_x * 0.0002645833) - self.image_center_X), 2)
            y_difference = round(((self.human_center_y * 0.0002645833) - self.image_center_Y), 2)
            rospy.loginfo(f"x_difference: {x_difference}, distance: {self.distance}")

            # Calculate angles using arctangent
            theta_x = (math.atan(x_difference / self.distance)) * 10
            theta_y = (math.atan(y_difference / self.distance)) * -10
            rospy.loginfo(f'x_theta: {theta_x:.2f}')
            
            self.angles.x, self.angles.y = round(theta_x, 2), round(theta_y, 2)
            self.angle_pub.publish(self.angles)

if __name__ == '__main__':
    try:
        camera_handler = CameraObject()
        while not rospy.is_shutdown():
            camera_handler.angleCalculation()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException as e:
        print('Exception:', e)
