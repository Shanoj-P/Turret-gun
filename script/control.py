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
        self.lower_base_pub = rospy.Publisher('/joint_base_controller/command', Float64, queue_size=0)
        rospy.Subscriber('/joint_states', JointState, self.readJointPosition)
        rospy.Subscriber('angle_topic', Point, self.turretControllerMain)

        # Initial positions
        self.turret_position = 0.0
        self.base_position = 0.0
        self.turret_base_position = 0.0
        self.previous_turret_angle  = 0.0
        self.previous_base_angle = 0.0
        # To store the commanded angles
        self.base_angle = Float64()
        self.turret_angle = Float64()

        # Rate to control the loop frequency
        self.rate = rospy.Rate(1)

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
            # rospy.loginfo(f"Current Joint Positions -> Turret: {self.turret_position}, Base: {self.base_position}, Turret Base: {self.turret_base_position}")
            pass
        else:
            rospy.logwarn("Failed to get joint properties!")

    def turretControllerMain(self,msg):
        self.base_angle= round((msg.x + self.turret_base_position),2)
        self.turret_angle= round((msg.y + self.turret_position*-1),2)
        if abs(self.turret_angle) - abs(self.turret_angle) < 0.05:
            self.turret_angle = self.previous_turret_angle
        # if abs(self.base_angle) - abs(self.previous_base_angle) < 0.01:
        #     self.base_angle = self.previous_base_angle
        # print(self.turret_base_position, self.turret_position, self.turret_base_position, msg.x, msg.y)

        if not math.isnan(self.base_angle) :
            if self.base_angle < 0:
                self.base_angle = 6.28 + self.base_angle 
            elif self.base_angle > 6.28:
                self.base_angle -= 6.28
                self.lower_base_angle = 0
                
            if self.base_angle > 3.14:
                self.lower_base_angle = round(((self.base_angle - 3.14) + self.turret_base_position),2)
                self.base_angle = 3.14 #round(self.base_angle - self.lower_base_angle,2)
                # self.lower_base_pub.publish(self.lower_base_angle)
                print("lower_base_angle: %2f" %self.lower_base_angle)
            else:
                self.lower_base_angle = 0
            self.lower_base_pub.publish(self.lower_base_angle)
            self.turret_pub.publish(self.turret_angle)
            self.base_pub.publish(self.base_angle)
            print('published')
            self.previous_turret_angle = self.turret_angle
            # self.previous_base_angle = self.base_angle

if __name__ == '__main__':
    try:
        turret_handler = turret_class()
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr(f"ROS Interrupt Exception: {e}")
