#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file (change filename accordingly)
csv_file = "joint_states.csv"
df = pd.read_csv(csv_file)

# Extract timestamps (convert from nanoseconds if needed)
df["time"] = df["%time"] * 1e-9  # Convert to seconds

# Extract joint data
joint_names = ["base_joint", "turret_base_joint", "turret_joint"]
positions = [f"field.position{i}" for i in range(len(joint_names))]
velocities = [f"field.velocity{i}" for i in range(len(joint_names))]
efforts = [f"field.effort{i}" for i in range(len(joint_names))]

# Function to show multiple graphs
def plot_graph(x, y_list, labels, xlabel, ylabel, title):
    plt.figure(figsize=(10, 5))
    for y, label in zip(y_list, labels):
        plt.plot(df[x], df[y], label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show(block=False)  # Allow multiple plots

# Plot Position over Time
plot_graph("time", positions, [f"{j} Position" for j in joint_names], "Time (s)", "Position (rad)", "Joint Positions Over Time")

# Plot Velocity over Time
plot_graph("time", velocities, [f"{j} Velocity" for j in joint_names], "Time (s)", "Velocity (rad/s)", "Joint Velocities Over Time")

# Plot Effort (Torque) over Time
plot_graph("time", efforts, [f"{j} Effort" for j in joint_names], "Time (s)", "Effort (Nm)", "Joint Effort Over Time")

# Phase Space: Position vs. Velocity
plt.figure(figsize=(10, 5))
for i, joint in enumerate(joint_names):
    plt.scatter(df[positions[i]], df[velocities[i]], label=f"{joint}")
plt.xlabel("Position (rad)")
plt.ylabel("Velocity (rad/s)")
plt.title("Phase Space: Position vs Velocity")
plt.legend()
plt.grid()
plt.show(block=False)  # Show without blocking

# Keep all plots open
plt.show()
