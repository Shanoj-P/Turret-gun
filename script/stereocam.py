#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class camera_object():
    def __init__(self):
        rospy.init_node("object_detection", anonymous=False)
        self.bridge = CvBridge()
        
        rospy.Subscriber('/left_camera1/image_raw', Image, self.left_image_callback)
        rospy.Subscriber('/right_camera1/image_raw', Image, self.right_image_callback)
        
        self.left_cv_image = None
        self.right_cv_image = None
        
        self.focal_length = 476.703  # in pixels, you need to know this from your camera calibration
        self.baseline = 0.0631


    def left_image_callback(self, msg):
        self.left_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # print("Left image received!")

    def right_image_callback(self, msg):
        self.right_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # print("Right image received!")

    def disparity_to_depth(self, disparity):
        if disparity is None:
            print("Error: Disparity map is None, skipping depth calculation.")
            return None
        
        # Avoid division by zero (handle very small disparity)
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid_disparity = disparity > 0.0  # Mask to exclude invalid disparity values
        
        # Depth calculation: Z = (focal_length * baseline) / disparity
        depth[valid_disparity] = (self.focal_length * self.baseline) / disparity[valid_disparity]
        return depth

    def stereocamera(self):
        if self.left_cv_image is None or self.right_cv_image is None:
            print("Error: Unable to load images.")
            return None  # Early exit if images are not yet available
            
        # Convert to grayscale
        left_gray = cv2.cvtColor(self.left_cv_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_cv_image, cv2.COLOR_BGR2GRAY)

        # Stereo BM algorithm for disparity calculation
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(left_gray, right_gray)

        # If disparity is None, print an error and return None
        if disparity is None:
            print("Error: Disparity calculation failed.")
            return None

        # Normalize the disparity map for better visualization
        disparity_normalized = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)

        # Show disparity map
        cv2.imshow('Disparity', disparity_normalized)
        cv2.waitKey(1)  # Use waitKey(1) for real-time update
        return disparity

    def displayDepthMap(self, depth_map):
        if depth_map is None:
            print("Error: Depth map is None, skipping display.")
            return
        
        # Normalize depth map for display
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_normalized = np.uint8(depth_map_normalized)
        cv2.imshow('Depth Map', depth_map_normalized)
        cv2.waitKey(1)


if __name__ == '__main__':
    try:
        camera_handler = camera_object()
        rate = rospy.Rate(10)
        rate.sleep()
        while not rospy.is_shutdown():
            # Wait for both images to be available before processing
            disparity = camera_handler.stereocamera()
            if disparity is not None:
                depth_map = camera_handler.disparity_to_depth(disparity)
                print(depth_map)
                if depth_map is not None:
                    camera_handler.displayDepthMap(depth_map)
            
            rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        pass
