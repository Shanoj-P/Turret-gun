#!/usr/bin/env python
import rospy
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CompressedImage
import cv2
import math
from geometry_msgs.msg import Point
import numpy as np
import torch

# class KalmanFilter:
#     def __init__(self, dt, std_acc, std_meas):
#         """
#         Kalman Filter initialization for a 2D position estimation.
#         """
#         self.dt = dt
        
#         # Initial state [x_position, y_position, x_velocity, y_velocity]
#         self.x = np.matrix([[0], [0], [0], [0]])
        
#         # State transition matrix
#         self.F = np.matrix([[1, 0, dt, 0],
#                             [0, 1, 0, dt],
#                             [0, 0, 1, 0],
#                             [0, 0, 0, 1]])

#         # Measurement matrix
#         self.H = np.matrix([[1, 0, 0, 0],
#                             [0, 1, 0, 0]])

#         # Process noise covariance
#         self.Q = np.matrix([[0.25 * dt**4, 0, 0.5 * dt**3, 0],
#                             [0, 0.25 * dt**4, 0, 0.5 * dt**3],
#                             [0.5 * dt**3, 0, dt**2, 0],
#                             [0, 0.5 * dt**3, 0, dt**2]]) * std_acc**2

#         # Measurement noise covariance
#         self.R = np.matrix([[std_meas**2, 0],
#                             [0, std_meas**2]])

#         # Initial covariance matrix
#         self.P = np.eye(self.F.shape[1])

#     def predict(self):
#         # Predict state and covariance
#         self.x = self.F * self.x
#         self.P = self.F * self.P * self.F.T + self.Q
#         return self.x[0:2]  # Return predicted position

#     def update(self, z):
#         # Kalman Gain
#         S = self.H * self.P * self.H.T + self.R
#         K = self.P * self.H.T * np.linalg.inv(S)

#         # Update state with measurement
#         y = np.matrix(z).T - self.H * self.x
#         self.x = self.x + K * y

#         # Update covariance
#         I = np.eye(self.H.shape[1])
#         self.P = (I - K * self.H) * self.P
#         return self.x[0:2]


class CameraObject:
    def __init__(self):
        rospy.init_node('human_detection', anonymous=False)
        self.bridge = CvBridge()
        self.model = YOLO('/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/runs/detect/train8/weights/best.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        
        # Image parameters
        self.image_center_X = (416 * 0.0002645833) / 2
        self.image_center_Y = (320 * 0.0002645833) / 2
        
        # Initialize variables
        self.angles = Point()
        self.distance = None
        self.human_center_x = None
        self.human_center_y = None
        self.angle_pub = rospy.Publisher('angle_topic', Point, queue_size=0)
        self.image_pub = rospy.Publisher('/live_image', CompressedImage, queue_size=0)
        
        # Initialize Kalman Filter
        # self.kalman_filter = KalmanFilter(dt=1.0, std_acc=1.0, std_meas=0.5)
        
        # Subscribers
        rospy.Subscriber('/left_camera_1/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/left_camera_1/depth/image_raw', Image, self.depth_callback)
    

    def image_callback(self, msg):
        # Process image and detect humans
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(cv_image)
        self.annotated_frame = results[0].plot()
        _, jpeg = cv2.imencode('.jpg', self.annotated_frame)
        ros_img = CompressedImage()
        ros_img.header = msg.header
        ros_img.format = "jpeg"
        ros_img.data = jpeg.tobytes()
        self.image_pub.publish(ros_img)
        
        for box in results[0].boxes:
            # YOLO bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Center of the bounding box (measurement)
            self.human_center_x = (x1 + x2) // 2
            self.human_center_y = (y1 + y2) // 2
            
            # Kalman filter prediction and update
            # self.kalman_filter.predict()
            # filtered_center = self.kalman_filter.update([measured_center_x, measured_center_y])
            # self.human_center_x, self.human_center_y = int(filtered_center[0, 0]), int(filtered_center[1, 0])
            
            rospy.loginfo(f"Filtered Position: ({self.human_center_x}, {self.human_center_y})")
        
        cv2.imshow('human_detection', self.annotated_frame)
        cv2.waitKey(1)


    def depth_callback(self, msg):
        # Depth image processing
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if self.human_center_x is not None and self.human_center_y is not None:
            self.distance = depth_image[self.human_center_x, self.human_center_y]
            rospy.loginfo(f"Distance: {self.distance}")


    def angle_calculation(self):
        # Calculate angles based on filtered human position
        if self.distance is not None and self.human_center_x is not None and self.human_center_y is not None:
            x_difference = round(((self.human_center_x * 0.0002645833) - self.image_center_X), 2)
            y_difference = round((self.image_center_Y - (self.human_center_y * 0.0002645833) ), 2)
            
            # Set small differences to zero to avoid micro-movements
            # x_difference = 0 if abs(x_difference) < 0.005 else x_difference
            # y_difference = 0 if abs(y_difference) < 0.005 else y_difference

            theta_x = math.asin(x_difference / self.distance) * 1.5
            theta_y = math.asin(y_difference / self.distance) * 1.5
            
            # Publish angles
            self.angles.x, self.angles.y = round(theta_x, 2), round(theta_y, 2)
            self.angle_pub.publish(self.angles)
            rospy.loginfo(f'Angle X: {theta_x}, Angle Y: {theta_y}')


if __name__ == '__main__':
    try:
        camera_handler = CameraObject()
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rospy.loginfo('Tracking...')
            camera_handler.angle_calculation()
            rate.sleep()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException as e:
        print('Exception:', e)
