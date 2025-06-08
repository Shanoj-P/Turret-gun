#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float64MultiArray
import numpy as np

class aimAnglePublisher():
    def __init__(self):
        self.Pc = None  # Position of target in camera frame
        self.Pw = None  # Position of target in Base frame
        self.Rcw = None
        # self.x, self.y, self.z, self.w = None  # initialize quaternion variables
        self.muzzle_velocity = 32.307 # Initialize muzzle velocity of the gun
        rospy.init_node('aim_angle_publisher',anonymous=False)
        rospy.Subscriber('/target_point', Point, self.targetPositionCallback)
        rospy.Subscriber('/imu_publisher',Quaternion, self.imuCallback)
        self.panAndTiltPublisher = rospy.Publisher('/pan_and_tilt', Float64MultiArray, queue_size=0)
        self.pan_and_tilt_angles = Float64MultiArray()
        self.previous_pan = 0
        self.previous_tilt_Compensated = 0


    def targetPositionCallback(self,msg):
        if msg is not None:
            self.Pc = np.array([msg.x, msg.y, msg.z])    # subscribe Position of target in camera frame
            self.Pc = np.array([msg.x, msg.y, msg.z]) / 1000.0  # Convert mm to meters
            # print('camera frame: ', self.Pc)
    

    def imuCallback(self, msg):
        x, y, z, w = msg.x, msg.y, msg.z, msg.w  # Subscribe quaternion values
        # norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
        # x, y, z, w = x / norm, y / norm, z / norm, w / norm

        if msg is not None:
            self.Rcw = np.array([
                    [2*(w**2 + x**2)-1,    2*(x*y - z*w),      2*(x*z + y*w)],
                    [2*(x*y + z*w),        2*(w**2 + y**2)-1,  2*(y*z - x*w)],            # Initialize rotation matrix
                    [2*(x*z - y*w),        2*(y*z + x*w),      2*(w**2+z**2)-1]
                ])
    

    def cameraToWorldFrame(self):
        if self.Rcw is not None and self.Pc is not None:
            self.Pw = self.Rcw @ self.Pc
            print('world frame:',self.Pw )
            # self.Pw = self.Pc


    def calculatePanAndTilt(self):
        if self.Pw is not None:


            pan = np.arctan2(self.Pw[0], self.Pw[2])    # Calculate pan angle
            pan = np.degrees(pan)                       # Convert radians to degrees

            tilt_raw = np.arctan2(self.Pw[1], np.sqrt(self.Pw[0]**2 + self.Pw[2]**2)) # Calculate the pan angle without considering ballistic effects

            # Compensate for projectile drop
            distance = np.sqrt(self.Pw[0]**2 + self.Pw[1]**2 + self.Pw[2]**2) # Straight line distance to target
            time_of_flight = distance / self.muzzle_velocity    # Calculate time of flight
            projectile_drop = 0.5 * 9.81 * time_of_flight**2
            Y_actual = self.Pw[1] + projectile_drop
            tilt_compensated = np.arctan2(Y_actual, np.sqrt(self.Pw[0]**2 + self.Pw[2]**2)) #Calculate tilt angle by addressing ballestic effects
            tilt_compensated = np.degrees(tilt_compensated)     # Convert tilt angle from radians to degrees
            if abs(pan) < 4:
                pan = 0 
            if abs(tilt_compensated) < 4:
                tilt_compensated = 0

            # if pan > 0:
            #     pan = min((90 + pan), 180)  # Limit pan angles within servo range 
            # elif pan < 0:
            #     pan = max((90 + pan), 0)    # Limit pan angles within servo range 
            # else :
            #     pan = 90
            # if tilt_compensated > 90:
            #     tilt_compensated = min((90 + tilt_compensated), 180)
            # elif tilt_compensated < 0:
            #     tilt_compensated = max((90 + tilt_compensated), 70)
            # else :
            #     tilt_compensated = 90
            self.pan_and_tilt_angles.data = [(pan*-1), tilt_compensated]
            self.panAndTiltPublisher.publish(self.pan_and_tilt_angles)
            self.Pc, self.Pw, self.Rcw = None, None, None



if __name__ == '__main__':
    angle_publisher = aimAnglePublisher()
    rate = rospy.Rate(1.5)
    while not rospy.is_shutdown():
        angle_publisher.cameraToWorldFrame()
        angle_publisher.calculatePanAndTilt()
        rate.sleep()
