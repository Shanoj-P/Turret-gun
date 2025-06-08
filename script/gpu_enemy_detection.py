#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO

# === Load stereo calibration parameters ===
cv_file = cv2.FileStorage("/home/shanoj/catkin_ws/src/turret_gun_description/script/stereo_params.xml", cv2.FILE_STORAGE_READ)
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

# === Focal length and baseline ===
focal_length_px = P1[0, 0]
baseline_mm = abs(T[0][0])
Fx = P1[0, 0]
Fy = P1[1, 1]
target_point =Point()

# === Load YOLOv8 model ===
model = YOLO('/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/runs/detect/train10/weights/best.pt').to('cuda')
# model.verbose = False

# === CV Bridge and Stereo matcher ===
bridge = CvBridge()
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# === Global variables ===
left_frame = None
right_frame = None

def left_callback(msg):
    global left_frame
    left_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def right_callback(msg):
    global right_frame
    right_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def main():
    global left_frame, right_frame

    rospy.init_node('stereo_depth_yolo_node', anonymous=True)

    rospy.Subscriber("/camera/left/image_raw", Image, left_callback)
    rospy.Subscriber("/camera/right/image_raw", Image, right_callback)
    target_point_publisher = rospy.Publisher("/target_point", Point, queue_size=1)

    rospy.loginfo("Waiting for camera frames...")
    while left_frame is None or right_frame is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    image_size = (left_frame.shape[1], left_frame.shape[0])
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    rospy.loginfo("Starting stereo + YOLOv8 processing...")
    prev_time = cv2.getTickCount()
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        if left_frame is None or right_frame is None:
            continue

        rectL = cv2.remap(left_frame, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(right_frame, map2x, map2y, cv2.INTER_LINEAR)

        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disparity_msg = bridge.cv2_to_imgmsg(disp_vis, encoding="mono8")
        disparity_msg.header.stamp = rospy.Time.now()
        # disparity_pub.publish(disparity_msg)

        # === YOLOv8 object detection ===
        results = model(rectL, verbose = False,device = 'cuda')[0]

        for box in results.boxes:
            if box.conf < 0.5:
                continue  # Skip low-confidence detections
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            targetX, targetY = (x1 + x2) // 2, (y1 + y2) // 2

            if 0 <= targetX < disparity.shape[1] and 0 <= targetY < disparity.shape[0]:
                Z = points_3D[targetY, targetX, 2]
                if np.isfinite(Z) and Z > 0:
                    label = f"{results.names[int(box.cls[0])]} {Z:.0f}mm"
                    print(f"Detected {label}")
                    cv2.rectangle(rectL, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rectL, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    h, w, _ = rectL.shape
                    Cx = w//2
                    Cy = h //2
                    X = ((targetX-Cx)*Z)/Fx
                    Y = ((targetY-Cy)*Z)/Fy
                    target_point.x, target_point.y, target_point.z = X,Y,Z
                    target_point_publisher.publish(target_point)
                else:
                    # rospy.logwarn_throttle(1, "Invalid depth at object center: ({}, {})".format(targetX, targetY))
                    pass
        

        # === FPS ===
        cur_time = cv2.getTickCount()
        time_diff = (cur_time - prev_time) / cv2.getTickFrequency()
        fps = 1 / time_diff
        prev_time = cur_time
        cv2.putText(rectL, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Left Rectified + Detections", rectL)
        cv2.imshow("Disparity", disp_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # rate.sleep()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
