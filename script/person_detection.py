#!/usr/bin/env python
import rospy
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import math
from geometry_msgs.msg import Point
import torch

class cameraObject:
    def __init__(self):
        rospy.init_node('human_detection', anonymous=False)
        self.bridge = CvBridge()
        self.model = YOLO('/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/runs/detect/train8/weights/best.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.image_center_X = (640 * 0.0002645833) / 2
        self.image_center_Y = (480 * 0.0002645833) / 2
        self.angles = Point()
        self.distance = None
        self.human_center_x = None
        self.human_center_y = None
        self.angle_pub = rospy.Publisher('angle_topic', Point, queue_size=0)
        rospy.Subscriber('/left_camera_1/color/image_raw', Image, self.imageCallback)
        rospy.Subscriber('/left_camera_1/depth/image_raw', Image, self.depthCallback)

        # rospy.Subscriber('/stereo/left/image_raw', Image, self.imageCallback)
        # rospy.Subscriber('/camera/depth/image_raw', Image, self.depthCallback)
    

    def imageCallback(self, msg):
        print('started')
        cvImage = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(cvImage)
        annotated_frame = results[0].plot()
        for box in results[0].boxes:
            # YOLO bounding boxes in the format [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Calculate the center of the bounding box
            self.human_center_x = (x1 + x2) // 2
            self.human_center_y = (y1 + y2) // 2
        cv2.imshow('human_detection', annotated_frame)
        cv2.waitKey(1)


    def depthCallback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if self.human_center_x is not None and self.human_center_y is not None:
            self.distance = depth_image[self.human_center_y, self.human_center_x] 
            print(self.distance)


    def angleCalculation(self):
        if self.distance is not None:
            
            x_difference = round(((self.human_center_x * 0.0002645833) - self.image_center_X),2)
            y_difference = round(((self.human_center_y * 0.0002645833) - self.image_center_Y),2)
            rospy.loginfo("x_difference %2f, distance %2f" %(x_difference, self.distance))
            # rospy.loginfo(y_difference)
            theta_x = (math.atan(x_difference / self.distance)) * 10 
            theta_y = (math.atan(y_difference / self.distance)) * -10
            rospy.loginfo('x thete: %2f' %theta_x)
            self.angles.x, self.angles.y = round(theta_x,2), round(theta_y,2)
            self.angle_pub.publish(self.angles)
    

if __name__ == '__main__':
    try:
        cameraHandler = cameraObject()
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            # rospy.loginfo('start')
            cameraHandler.angleCalculation()
            rate.sleep()
            # if cameraHandler.distance is not None:
            #     rospy.loginfo("distance: %.2f" % cameraHandler.distance)
            # else:
            #     rospy.logwarn("distance not available yet")
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException as e:
        print('exception',e)
