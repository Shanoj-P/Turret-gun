#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox


class CameraObject:
    def __init__(self):
        rospy.init_node("object_detection", anonymous=False)
        self.bridge = CvBridge()

        # Subscribe to the left image and disparity topics
        rospy.Subscriber('/stereo/left/image_color', Image, self.left_image_callback)
        rospy.Subscriber('/stereo/disparity', DisparityImage, self.disparity_image_callback)

        self.left_image = None
        self.disparity_image = None
        self.bbox = None
        self.label = None
        self.conf = None
        self.focal_length = 500  # Example focal length in pixels (adjust based on your camera)
        self.baseline = 0.1  # Example baseline in meters (adjust based on your stereo setup)

    def left_image_callback(self, msg):
        # Convert the left camera image to OpenCV format
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Detect objects in the image
        self.bbox, self.label, self.conf = cv.detect_common_objects(self.left_image)
        self.object_center()

    def disparity_image_callback(self, msg):
        # Convert the disparity image from stereo_msgs/DisparityImage to OpenCV format
        try:
            # The image field in the DisparityImage message contains the actual disparity data
            self.disparity_image = self.bridge.imgmsg_to_cv2(msg.image, "32FC1")
            if self.disparity_image is not None:
                cv2.imshow("Disparity Image", self.disparity_image)
                cv2.waitKey(1)
                

        except Exception as e:
            rospy.logerr(f"Failed to convert disparity image: {e}")

    def object_center(self):
        if len(self.bbox) > 0:
            # Example: Track the first detected object
            x_min, y_min, x_max, y_max = self.bbox[0]
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            print(f"Object center: ({center_x}, {center_y})")
            disparity = self.disparity_image[center_y, center_x]
            print(f"Disparity at object center: {disparity}")


            # Now use the disparity image to get the 3D position of the object
            if self.disparity_image is not None:
                depth_value = self.get_depth_at(center_x, center_y)
                print(f"Depth at object center: {depth_value}")

                # Convert to 3D world position
                object_pos_3d = self.image_to_world(center_x, center_y, depth_value)
                print(f"Object 3D position: {object_pos_3d}")

                # Move the manipulator based on object position
                # self.move_manipulator(object_pos_3d)

    def get_depth_at(self, x, y):
        # Calculate the depth from the disparity (inverse of disparity)
        disparity = self.disparity_image[y, x]
        if disparity > 0:
            depth = (self.focal_length * self.baseline) / disparity  # Convert disparity to depth (in meters)
        else:
            depth = 0
        return depth

    def image_to_world(self, x, y, depth):
        # Convert (x, y) in image space to 3D world space using camera intrinsics
        fx = self.focal_length  # Focal length in x (example)
        fy = self.focal_length  # Focal length in y (example)
        cx = 320  # Principal point x (example)
        cy = 240  # Principal point y (example)

        # Simple pinhole camera model
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z])

    # def move_manipulator(self, object_pos_3d):
    #     # Assuming simple joint movement logic based on object position
    #     joint_state = JointState()
    #     joint_state.name = ['base_joint', 'shoulder_joint', 'elbow_joint']
    #     joint_state.position = [0.0, 0.0, 0.0]  # Dummy values for now
    #     self.joint_pub.publish(joint_state)

if __name__ == '__main__':
    try:
        camera_handler = CameraObject()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
