#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import GetJointProperties
from sensor_msgs.msg import JointState  # Correct import
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
import math

class turret_class:

    def __init__(self):
        rospy.init_node('turret_controller_node', anonymous=False)
        self.turret_pub = rospy.Publisher('/joint_turret_controller/command', Float64, queue_size=0)
        self.base_pub = rospy.Publisher('/joint_turret_base_controller/command', Float64, queue_size=0)
        rospy.Subscriber('/joint_states', JointState, self.readJointPosition)
        rospy.Subscriber('angle_topic', Point, self.turretControllerMain)

        # Initial positions
        self.turret_position = 0.0
        self.base_position = 0.0
        self.turret_base_position = 0.0

        # To store the commanded angles
        self.base_angle = Float64()
        self.turret_angle = Float64()

        # Previous filtered angles for EMA
        self.prev_base_angle = 0.0
        self.prev_turret_angle = 0.0

        # Smoothing factor for EMA
        self.alpha = 0.3  # Light smoothing factor
        # Threshold for applying EMA filtering
        self.filter_threshold = 0.05  # Only filter small fluctuations

        # Rate to control the loop frequency
        self.rate = rospy.Rate(1)

    def exponential_moving_average(self, new_value, prev_value):
        return self.alpha * new_value + (1 - self.alpha) * prev_value

    def get_joint_properties(self, joint_name):
        rospy.wait_for_service('/gazebo/get_joint_properties')

        try:
            get_joint_properties = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)

            response = get_joint_properties(joint_name)
            
            return response.position[0]  

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None

    def readJointPosition(self, joint):
        self.turret_position = self.get_joint_properties("turret_joint")
        self.base_position = self.get_joint_properties("base_joint")
        self.turret_base_position = self.get_joint_properties("turret_base_joint")

        if self.turret_position is not None and self.base_position is not None and self.turret_base_position is not None:
            pass
        else:
            rospy.logwarn("Failed to get joint properties!")

    def turretControllerMain(self, msg):
        # Calculate raw angles
        raw_base_angle = round((msg.x + self.turret_base_position), 2)
        raw_turret_angle = round((msg.y + self.turret_position * -1), 2)

        # Apply conditional EMA filtering based on small fluctuation threshold
        if abs(raw_base_angle - self.prev_base_angle) < self.filter_threshold:
            self.base_angle = self.exponential_moving_average(raw_base_angle, self.prev_base_angle)
        else:
            self.base_angle = raw_base_angle  # Directly set if the change is large

        if abs(raw_turret_angle - self.prev_turret_angle) < self.filter_threshold:
            self.turret_angle = self.exponential_moving_average(raw_turret_angle, self.prev_turret_angle)
        else:
            self.turret_angle = raw_turret_angle  # Directly set if the change is large

        # Update previous values for next iteration
        self.prev_base_angle = self.base_angle
        self.prev_turret_angle = self.turret_angle

        print(self.turret_base_position, self.turret_position, msg.x, msg.y)

        if not math.isnan(self.base_angle):
            self.turret_pub.publish(Float64(self.turret_angle))
            self.base_pub.publish(Float64(self.base_angle))
            print('published')

if __name__ == '__main__':
    try:
        turret_handler = turret_class()
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr(f"ROS Interrupt Exception: {e}")
