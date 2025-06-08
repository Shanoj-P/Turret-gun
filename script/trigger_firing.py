#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
import sys
import termios
import tty

def get_key():
    """Get keyboard input (non-blocking)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def trigger_firing():
    rospy.init_node('trigger_firing', anonymous=False)
    trigger_publisher = rospy.Publisher('/trigger_firing', Int32, queue_size=1)
    rate = rospy.Rate(10)
    trigger = Int32()
    while not rospy.is_shutdown():
        key = get_key()
        if key == 'f':
            trigger.data = 1
        if key == 'q':
            break
        trigger_publisher.publish(trigger)
        trigger.data = 0
        rate.sleep()
    


if __name__ == '__main__':
    try:
        trigger_firing()
    except rospy.ROSInterruptException:
        pass