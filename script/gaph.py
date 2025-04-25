#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import GetJointProperties
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
import math
import matplotlib.pyplot as plt
import threading
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # Non-interactive backend

class turret_class:

    def __init__(self):
        rospy.init_node('turret_controller_node', anonymous=False)
        self.turret_pub = rospy.Publisher('/joint_turret_controller/command', Float64, queue_size=0)
        self.base_pub = rospy.Publisher('/joint_turret_base_controller/command', Float64, queue_size=0)
        self.lower_base_pub = rospy.Publisher('/joint_base_controller/command', Float64, queue_size=0)
        rospy.Subscriber('/joint_states', JointState, self.readJointPosition)
        rospy.Subscriber('angle_topic', Point, self.turretControllerMain)

        # Real-time data storage
        self.x_data, self.y_data = [], []
        self.start_time = time.time()

        # Thread-safe locks
        self.lock = threading.Lock()

        # Start the plotting thread
        self.plot_thread = threading.Thread(target=self.plot_data)
        self.plot_thread.daemon = True
        self.plot_thread.start()

        # Initial positions
        self.turret_position = 0.0
        self.base_position = 0.0
        self.turret_base_position = 0.0
        self.previous_turret_angle = 0.0
        self.previous_base_angle = 0.0

        # To store the commanded angles
        self.base_angle = Float64()
        self.turret_angle = Float64()

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

    def turretControllerMain(self, msg):
        self.base_angle = round((msg.x + self.turret_base_position), 2)
        self.turret_angle = round((msg.y + self.turret_position * -1), 2)

        if abs(self.turret_angle - self.previous_turret_angle) < 0.05:
            self.turret_angle = self.previous_turret_angle

        self.previous_turret_angle = self.turret_angle

        # Update data for plotting
        with self.lock:
            current_time = time.time() - self.start_time
            self.x_data.append(current_time)
            self.y_data.append(self.turret_angle)

    def plot_data(self):
        plt.switch_backend("Agg")  # Use non-GUI backend
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)  # For rendering with Agg
        ax.set_title("Real-Time Turret Angle")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Turret Angle")
        ax.grid()

        while not rospy.is_shutdown():
            with self.lock:
                ax.clear()
                ax.plot(self.x_data, self.y_data, label="Turret Angle")
                ax.legend()
                ax.grid()

                # Save the plot as an image
                fig.savefig("/tmp/realtime_plot.png")

            rospy.sleep(0.5)  # Plot every 0.5 seconds

if __name__ == '__main__':
    try:
        turret_handler = turret_class()
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr(f"ROS Interrupt Exception: {e}")
