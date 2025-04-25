#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cvlib.object_detection import draw_bbox
import cv2
import cvlib as cv


class camera_object():
    def __init__(self):
        rospy.init_node("object_detection", anonymous=False)
        self.bridge = CvBridge()
        
        rospy.Subscriber('/left_camera_1/color/image_raw', Image, self.image_callback)
        self.bbox = None


    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        self.bbox, self.label, self.conf = cv.detect_common_objects(cv_image)
        self.output = draw_bbox(cv_image, self.bbox,self.label, self.conf)
        cv2.imshow('object_detection', self.output)
        cv2.waitKey(1)

    def track_object(self):
        if len(self.bbox[0]) > 3:
            center_x = (self.bbox[0][0] + self.bbox[0][2]) / 2
            center_y = (self.bbox[0][1] + self.bbox[0][3]) / 2
            print(center_x, center_y)
if __name__ == '__main__':
    try:
        camera_handler = camera_object()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if camera_handler.bbox != None:
                camera_handler.track_object()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        pass