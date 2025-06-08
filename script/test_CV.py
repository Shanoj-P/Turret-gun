#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# === Load stereo calibration parameters ===
cv_file = cv2.FileStorage("stereo_params.xml", cv2.FILE_STORAGE_READ)

K1 = cv_file.getNode("K1").mat()
D1 = cv_file.getNode("D1").mat()
K2 = cv_file.getNode("K2").mat()
D2 = cv_file.getNode("D2").mat()
R = cv_file.getNode("R").mat()
T = cv_file.getNode("T").mat()
R1 = cv_file.getNode("R1").mat()
R2 = cv_file.getNode("R2").mat()
P1 = cv_file.getNode("P1").mat()
P2 = cv_file.getNode("P2").mat()
Q = cv_file.getNode("Q").mat()

cv_file.release()

# === Extract focal length and baseline ===
focal_length_px = P1[0, 0]
baseline_mm = abs(T[0][0])
print("Focal length (px): {}".format(focal_length_px))
print("Baseline (mm): {}".format(baseline_mm))

# === Initialize CV Bridge ===
bridge = CvBridge()

# === Global variables ===
left_frame = None
right_frame = None

# === Stereo SGBM matcher ===
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # Must be divisible by 16
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

def left_callback(msg):
    global left_frame
    left_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def right_callback(msg):
    global right_frame
    right_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def main():
    global left_frame, right_frame

    rospy.init_node('stereo_depth_node', anonymous=True)

    # === Subscribers ===
    rospy.Subscriber("/camera/left/image_raw", Image, left_callback)
    rospy.Subscriber("/camera/right/image_raw", Image, right_callback)

    # === Publisher for disparity ===
    disparity_pub = rospy.Publisher("/stereo/disparity", Image, queue_size=1)

    # === Wait until frames are received ===
    rospy.loginfo("Waiting for camera frames...")
    while left_frame is None or right_frame is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # === Create rectification maps ===
    image_size = (left_frame.shape[1], left_frame.shape[0])
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    prev_time = cv2.getTickCount()

    rospy.loginfo("Starting depth estimation...")

    while not rospy.is_shutdown():
        if left_frame is None or right_frame is None:
            continue

        # === Rectify frames ===
        rectL = cv2.remap(left_frame, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(right_frame, map2x, map2y, cv2.INTER_LINEAR)

        # === Convert to grayscale ===
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        # === Compute disparity map ===
        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

        # === Reproject to 3D ===
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # === Normalize disparity for visualization ===
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        # === Publish disparity image ===
        disparity_msg = bridge.cv2_to_imgmsg(disp_vis, encoding="mono8")
        disparity_msg.header.stamp = rospy.Time.now()
        disparity_pub.publish(disparity_msg)

        # === Display center pixel depth ===
        h, w = disparity.shape
        disp_center = disparity[h // 2, w // 2]

        if disp_center > 0:
            Z = points_3D[h // 2, w // 2, 2]
            rospy.loginfo_throttle(1, "Center disparity: {:.2f}, Depth (Z): {:.2f} mm".format(disp_center, Z))
        else:
            rospy.logwarn_throttle(1, "Invalid disparity at center pixel.")

        # === Visual overlay on disparity image ===
        cv2.circle(disp_vis, (w // 2, h // 2), 5, (255, 0, 0), 2)

        # === FPS Calculation ===
        cur_time = cv2.getTickCount()
        time_diff = (cur_time - prev_time) / cv2.getTickFrequency()
        fps = 1 / time_diff
        prev_time = cur_time

        # === Display FPS ===
        cv2.putText(rectL, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # === Show windows ===
        cv2.imshow("Left Rectified", rectL)
        cv2.imshow("Right Rectified", rectR)
        cv2.imshow("Disparity", disp_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
