#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cvlib.object_detection import draw_bbox
import cv2
import cvlib as cv
from std_msgs.msg import Float64
import numpy as np
from sensor_msgs.msg import JointState

class camera_object():
    def __init__(self):
        rospy.init_node("object_detection", anonymous=False)
        self.bridge = CvBridge()
        
        # Subscribe to the camera topic to get the raw images
        rospy.Subscriber('/left_camera1/image_raw', Image, self.image_callback)
        
        # Initialize object variables
        self.bbox = None
        self.center_x = None
        self.center_y = None
        
        # Publishers for the robot's joint position commands
        self.base_joint_pub = rospy.Publisher('/joint_turret_base_controller/command', Float64, queue_size=10)
        self.turret_joint_pub = rospy.Publisher('/joint_turret_controller/command', Float64, queue_size=10)
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        
        # Set the desired position of the object in the image (center of the camera)
        self.desired_center_x = 320  # Assuming 640x480 image, center at 320
        self.desired_center_y = 240  # Assuming 640x480 image, center at 240
        self.threshold = 5  # Tolerance for error


    def joint_state_callback(self, msg):
        joint_names = msg.name
        self.joint_positions = msg.position

        



    def image_callback(self, msg):
        # Convert the ROS image message to a OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Detect objects in the image
        self.bbox, self.label, self.conf = cv.detect_common_objects(cv_image)
        
        # Draw bounding boxes on the image
        self.output = draw_bbox(cv_image, self.bbox, self.label, self.conf)
        
        # Show the processed image
        cv2.imshow('object_detection', self.output)
        cv2.waitKey(1)
        
        if self.bbox:
            # Get the first bounding box (assuming we track the first object)
            self.center_x, self.center_y = self.calculate_center(self.bbox[0])
            
            # Call the track_object method to control robot movement
            self.track_object(self.center_x, self.center_y)

    def calculate_center(self, bbox):
        # Calculate the center of the bounding box (x1, y1, x2, y2)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return center_x, center_y

    def track_object(self, center_x, center_y):
        # Calculate the error in the X and Y direction
        error_x = self.desired_center_x - center_x
        error_y = self.desired_center_y - center_y
        print(error_x, error_y)
        # Initialize joint position commands for base and turret
        base_joint_position = 0
        turret_joint_position = 0
        
        # If error in X is outside threshold, move turret base
        if error_x > self.threshold or error_x < -self.threshold:
            # Map error in X to a suitable base joint position command (in radians)
            base_joint_position = (error_x * -0.001)  # Example scaling factor
            print('move')

        # If error in Y is outside threshold, move turret joint
        if error_y > self.threshold or error_y < -self.threshold:
            # Map error in Y to a suitable turret joint position command (in radians)
            turret_joint_position = (error_y * -0.001)  # Example scaling factor
        
        # Publish the new positions to the joints
        self.base_joint_pub.publish(base_joint_position)
        self.turret_joint_pub.publish(turret_joint_position)

if __name__ == '__main__':
    try:
        # Create the camera object handler
        camera_handler = camera_object()
        
        # Set the rate of the loop
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            rospy.spin()
        
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
        
    except rospy.ROSInterruptException:
        pass
