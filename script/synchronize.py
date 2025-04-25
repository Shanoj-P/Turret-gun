#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer

def callback(left_image, right_image, left_info, right_info):
    # Process the synchronized stereo images and camera info
    pass

def main():
    rospy.init_node('stereo_sync_node')

    # Create subscribers for left and right camera topics
    left_image_sub = Subscriber("/left_camera1/image_raw", Image)
    right_image_sub = Subscriber("/right_camera1/image_raw", Image)
    left_info_sub = Subscriber("/left_camera1/camera_info", CameraInfo)
    right_info_sub = Subscriber("/right_camera1/camera_info", CameraInfo)

    # Synchronize messages from both cameras
    ats = ApproximateTimeSynchronizer([left_image_sub, right_image_sub, left_info_sub, right_info_sub], queue_size=10, slop=0.1)
    ats.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    main()
