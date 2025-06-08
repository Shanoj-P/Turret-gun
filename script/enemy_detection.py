#!/usr/bin/env python
import rospy
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import math
from geometry_msgs.msg import Point
import torch
import time  # <- added for FPS calculation

class cameraObject:
    def __init__(self):
        rospy.init_node('human_detection', anonymous=False)
        self.bridge = CvBridge()
        self.model = YOLO('/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/runs/detect/train8/weights/best.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(self.device)
        self.image_center_X = (640 * 0.02645833) / 2
        self.image_center_Y = (480 * 0.02645833) / 2
        self.angles = Point()
        self.distance = None
        self.human_center_x = None
        self.human_center_y = None
        self.prev_time = time.time()  # <- initialize previous time

        # Read stereo calibration parameters
        cv_file = cv2.FileStorage("/home/shanoj/catkin_ws/src/turret_gun_description/script/stereo_params.xml", cv2.FILE_STORAGE_READ)
        self.K1 = cv_file.getNode("K1").mat()
        self.D1 = cv_file.getNode("D1").mat()
        self.K2 = cv_file.getNode("K2").mat()
        self.D2 = cv_file.getNode("D2").mat()
        self.R1 = cv_file.getNode("R1").mat()
        self.R2 = cv_file.getNode("R2").mat()
        self.P1 = cv_file.getNode("P1").mat()
        self.P2 = cv_file.getNode("P2").mat()
        self.Q = cv_file.getNode("Q").mat()
        cv_file.release()

        self.angle_pub = rospy.Publisher('angle_topic', Point, queue_size=0)
        rospy.Subscriber('/camera/left/image_raw', Image, self.imageCallback)
        rospy.Subscriber('/stereo/disparity', Image, self.depthCallback)

    def imageCallback(self, msg):
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time

        cvImage = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(cvImage, device = self.device,verbose=False)
        annotated_frame = cvImage.copy()

        for box in results[0].boxes:
            confidence = box.conf[0]
            if confidence > 0.50:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                self.human_center_x = (x1 + x2) // 2
                self.human_center_y = (y1 + y2) // 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # print(f"Detected object with confidence {confidence*100:.2f}% at ({self.human_center_x}, {self.human_center_y})")
        
        # Draw FPS on frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('human_detection', annotated_frame)
        cv2.waitKey(1)

    def depthCallback(self, msg):
        if self.human_center_x is not None and self.human_center_y is not None:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            points_3D = cv2.reprojectImageTo3D(depth_image, self.Q)
            center_disp = depth_image[self.human_center_y, self.human_center_x]
            if center_disp > 0:
                self.distance = points_3D[self.human_center_y, self.human_center_x, 2]
                rospy.loginfo(f"Center disparity: {center_disp:.2f}, Depth (Z): {self.distance:.2f} mm")
            else:
                rospy.loginfo("Invalid disparity at center pixel.")

    def angleCalculation(self):
        if self.distance is not None:
            x_difference = round(((self.human_center_x * 0.02645833) - self.image_center_X), 2)
            y_difference = round(((self.human_center_y * 0.02645833) - self.image_center_Y), 2)
            rospy.loginfo("x_difference %2f, distance %2f" % (x_difference, self.distance))
            theta_x = (math.atan(x_difference / self.distance)) 
            theta_y = (math.atan(y_difference / self.distance)) # *-1
            rospy.loginfo('x theta: %2f' % theta_x)
            self.angles.x, self.angles.y = theta_x, theta_y
            self.angle_pub.publish(self.angles)

if __name__ == '__main__':
    try:
        cameraHandler = cameraObject()
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            cameraHandler.angleCalculation()
            rate.sleep()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException as e:
        print('exception', e)
