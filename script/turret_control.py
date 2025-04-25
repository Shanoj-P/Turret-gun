#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import math



class turret_class():
    

    def __init__(self):
        rospy.init_node('turret_controller_node', anonymous=False)
        self.turret_pub = rospy.Publisher('/joint_turret_controller/command', Float64, queue_size=0)
        self.base_pub = rospy.Publisher('/joint_turret_base_controller/command', Float64, queue_size=0)
        rospy.Subscriber('/joint_states', JointState, self.readJointPosition)
        rospy.Subscriber('angle_topic', Point, self.turretControllerMain)
        self.base_angle = Float64()
        self.turret_angle = Float64()
        self.rate = rospy.Rate(1)


    def readJointPosition(self, joint):
        self.turret_position = joint.position[0]
        self.base_position = joint.position[1]
        self.turret_base_position = joint.position[2]
        
    
    def turretControllerMain(self,msg):
        self.base_angle.data= round((msg.x),2)
        self.turret_angle.data= round((msg.y),2)
        print(self.turret_base_position, self.turret_position)

        # if not math.isnan(self.base_angle) :
        #     self.turret_pub.publish(self.turret_angle)
        #     self.base_pub.publish(self.base_angle)
        #     print('published')

    def turretControllerMain(self,msg):
        self.base_angle= round((msg.x + self.turret_base_position),2)
        self.turret_angle= round((msg.y + self.turret_position),2)
        print(self.turret_base_position, self.turret_position)

        if not math.isnan(self.base_angle) :
            self.turret_pub.publish(self.turret_angle)
            self.base_pub.publish(self.base_angle)
            print('published')
            self.rate.sleep()
        

if __name__ == '__main__':
    try:
        turret_handler = turret_class()
        # rate = rospy.Rate(1)
        # while not rospy.is_shutdown():
        #     turret_handler.move_manipulator
        #     rate.sleep()
        rospy.spin()
    except rospy.ROSInterruptException as e:
        print(e)