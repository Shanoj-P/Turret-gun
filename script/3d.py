#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class VisualizeDepth(object):
    def __init__(self):
        rospy.init_node('visualize_depth')
        self.bridge = CvBridge()
        self.disparity_sub = rospy.Subscriber('/stereo/disparity', DisparityImage, self.disparity_callback)
        self.depth_pub = rospy.Publisher('/depth_image_colored', Image, queue_size=1)

        self.focal_length = 871.22  # Your focal length in pixels
        self.baseline = 0.097        # Your baseline in meters (replace with the correct value)
        self.min_depth = 0.5       # Adjust based on your scene
        self.max_depth = 5.0       # Adjust based on your scene

    def disparity_callback(self, msg):
        try:
            disparity_img = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="32FC1")
            depth_img = np.zeros_like(disparity_img)

            for v in range(disparity_img.shape[0]):
                for u in range(disparity_img.shape[1]):
                    disparity = disparity_img[v, u]
                    if disparity > 0:
                        depth = self.focal_length * self.baseline / disparity
                        depth_img[v, u] = depth
                    else:
                        depth_img[v, u] = 0.0

            # Normalize depth image to 0-255 range
            normalized_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Apply a colormap
            colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_VIRIDIS)

            # Convert back to ROS Image message
            depth_msg = self.bridge.cv2_to_imgmsg(colored_depth, encoding="bgr8")
            depth_msg.header = msg.header
            self.depth_pub.publish(depth_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")

if __name__ == '__main__':
    try:
        visualizer = VisualizeDepth()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass