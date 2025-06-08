#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('/home/shanoj/Downloads/best.pt')

class YoloRosNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('yolo_human_detector')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/left/image_raw', Image, self.image_callback)
        
        # Optionally: set up annotated image publisher
        self.image_pub = rospy.Publisher('/right/image_annotated', Image, queue_size=1)
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        # Perform YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Show the annotated frame
        cv2.imshow("YOLO Detection", annotated_frame)
        cv2.waitKey(1)

        # (Optional) Publish the annotated image back to ROS
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.image_pub.publish(annotated_msg)
        except Exception as e:
            rospy.logerr(f"Error converting/publishing annotated image: {e}")

    def run(self):
        rospy.loginfo("YOLO Human Detector Node Started.")
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = YoloRosNode()
    node.run()