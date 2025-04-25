#!/usr/bin/env python
import rospy
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pc2
import cv2
import math


class CameraObject:
    def __init__(self):
        rospy.init_node('human_detection', anonymous=False)
        self.bridge = CvBridge()
        self.model = YOLO('/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/runs/detect/train8/weights/best.pt')

        self.image_width = 640
        self.image_height = 480
        self.image_center_X = (self.image_width * 0.0002645833) / 2
        self.image_center_Y = (self.image_height * 0.0002645833) / 2

        self.angles = Point()
        self.distance = None
        self.human_center_x = None
        self.human_center_y = None

        self.angle_pub = rospy.Publisher('angle_topic', Point, queue_size=10)
        rospy.Subscriber('/stereo/left/image_rect_color', Image, self.imageCallback)
        rospy.Subscriber('/stereo/points2', PointCloud2, self.depthCallback)

    def imageCallback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        results = self.model(cv_image, verbose = False)  # height, width
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            self.human_center_x = (x1 + x2) // 2
            self.human_center_y = (y1 + y2) // 2
            break  # Only take the first detection

        cv2.imshow('Human Detection', annotated_frame)
        cv2.waitKey(1)

    def depthCallback(self, cloud_msg):
        if self.human_center_x is not None and self.human_center_y is not None:
            u, v = self.human_center_x, self.human_center_y
            gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=False, uvs=[(u, v)])
            point = next(gen, None)
            x, y, z = point
            rospy.loginfo("x: %.2f, y: %.2f, z: %.2f",x,y,z)

            # if point and not any(map(math.isnan, point)):
            #     x, y, z = point
            #     self.distance = math.sqrt(x**2 + y**2 + z**2)
            #     theta_x, theta_y = self.calculateAngles(x, y, z)
            #     rospy.loginfo("Distance: %.2f m, Angle X: %.2f°, Angle Y: %.2f°, d_z: %2f", self.distance, theta_x, theta_y,z)

            #     self.angles.x = theta_x
            #     self.angles.y = theta_y
            #     self.angles.z = self.distance
            #     self.angle_pub.publish(self.angles)
            # else:
            #     rospy.logwarn("Invalid point at (%d, %d)", u, v)

    def calculateAngles(self, x, y, z):
        pixel_size = 0.0002645833  # meters per pixel (assumed from image size and FOV)
        x_pixel_diff = (self.human_center_x * pixel_size) - self.image_center_X
        y_pixel_diff = (self.human_center_y * pixel_size) - self.image_center_Y
        theta_x = math.degrees(math.atan2(x_pixel_diff, z))
        theta_y = -math.degrees(math.atan2(y_pixel_diff, z))
        return round(theta_x, 2), round(theta_y, 2)


if __name__ == '__main__':
    try:
        camera_handler = CameraObject()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException as e:
        print('Exception:', e)
