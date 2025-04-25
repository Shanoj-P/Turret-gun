#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64

def control_turret():
    # Initialize the ROS node
    rospy.init_node('turret_controller_node', anonymous=True)

    # Publisher for the /joint_turret_controller/command topic
    turret_pub = rospy.Publisher('/joint_turret_controller/command', Float64, queue_size=10)

    # Define the rate at which we want to publish messages (e.g., 10 Hz)
    rate = rospy.Rate(10)

    # Example position commands to send
    position_sequence = [0.0, 0.5, 1.0, -0.5, -1.0, 0.0]  # in radians or degrees depending on your setup
    position_index = 0

    # Wait a bit for the publisher to register with ROS
    rospy.sleep(1)

    while not rospy.is_shutdown():
        # Set the target position for the turret
        target_position = Float64()
        target_position.data = position_sequence[position_index]

        # Publish the position command
        rospy.loginfo(f"Setting turret position to: {target_position.data}")
        turret_pub.publish(target_position)

        # Increment position index, reset if it exceeds the sequence length
        position_index = (position_index + 1) % len(position_sequence)

        # Wait for the next loop
        rate.sleep()

if __name__ == '__main__':
    try:
        control_turret()
    except rospy.ROSInterruptException:
        pass
