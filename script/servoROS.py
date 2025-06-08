#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16MultiArray
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

def main():
    rospy.init_node('servo_keyboard_control')
    pub = rospy.Publisher('/servo_angles', Int16MultiArray, queue_size=10)
    
    angle1 = 90
    angle2 = 90
    trigger = 0
    step = 5

    print("Control servo1 with 'a' (left) and 'd' (right)")
    print("Control servo2 with 'w' (up) and 's' (down)")
    print("Press 'q' to quit.")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        key = get_key()
        if key == 'a':
            angle1 = max(0, angle1 - step)
        elif key == 'd':
            angle1 = min(180, angle1 + step)
        elif key == 'w':
            angle2 = min(180, angle2 + step)
        elif key == 's':
            angle2 = max(0, angle2 - step)
        elif key == 'f':
            trigger = 1
        elif key == 'n':
            trigger = 0
        elif key == 'q':
            break

        msg = Int16MultiArray()
        msg.data = [angle1, angle2, trigger]
        pub.publish(msg)

        print(f"Servo1: {angle1}, Servo2: {angle2}")
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
