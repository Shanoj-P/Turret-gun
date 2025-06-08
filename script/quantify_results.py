import rosbag
import numpy as np
import matplotlib.pyplot as plt

OBJECT_NAME = "human_without_hand"
ARM_LINK_NAME = "turret_gun::turret_arm_1"
BAG_FILE = "/home/shanoj/2025-05-15-20-59-22.bag"
ALIGN_AXIS = 1  # 0: X, 1: Y, 2: Z
# turret_gun::rotating_base_1
#turret_gun::turret_base_1
enemy_pos = []
arm_pos = []
times = []

with rosbag.Bag(BAG_FILE) as bag:
    obj_pos = None
    arm_p = None
    t_obj = None
    t_arm = None
    for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states', '/gazebo/link_states']):
        if topic == '/gazebo/model_states' and OBJECT_NAME in msg.name:
            idx = msg.name.index(OBJECT_NAME)
            pos = msg.pose[idx].position
            obj_pos = np.array([pos.x, pos.y, pos.z])
            t_obj = t.to_sec()
        elif topic == '/gazebo/link_states' and ARM_LINK_NAME in msg.name:
            idx = msg.name.index(ARM_LINK_NAME)
            pos = msg.pose[idx].position
            arm_p = np.array([pos.x, pos.y, pos.z])
            t_arm = t.to_sec()
        if obj_pos is not None and arm_p is not None:
            times.append(max(t_obj, t_arm))
            enemy_pos.append(obj_pos)
            arm_pos.append(arm_p)
            obj_pos = None
            arm_p = None

enemy_pos = np.array(enemy_pos)
arm_pos = np.array(arm_pos)
times = np.array(times)

if len(enemy_pos) and len(arm_pos) and enemy_pos.shape == arm_pos.shape:
    # Axis-aligned error (absolute difference)
    axis_error = np.abs(arm_pos[:, ALIGN_AXIS] - enemy_pos[:, ALIGN_AXIS])

    print(f"Alignment error along axis {ALIGN_AXIS} (0=x,1=y,2=z):")
    print("Mean error:", np.mean(axis_error))
    print("Minimum error:", np.min(axis_error))
    print("Maximum error:", np.max(axis_error))

    # Plot error over time
    plt.figure()
    plt.plot(times, axis_error)
    plt.xlabel('Time [s]')
    axis_names = ['X', 'Y', 'Z']
    plt.ylabel(f'Alignment Error [{axis_names[ALIGN_AXIS]}] (m)')
    plt.title(f'Straight Line Alignment Error ({axis_names[ALIGN_AXIS]})')
    plt.grid()
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data. Check names and bag file.")