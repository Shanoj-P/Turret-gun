#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
from threading import Lock

# === Load stereo calibration parameters ===
cv_file = cv2.FileStorage("/home/shanoj/catkin_ws/src/turret_gun_description/script/stereo_params.xml", cv2.FILE_STORAGE_READ)
K1, D1 = cv_file.getNode("K1").mat(), cv_file.getNode("D1").mat()
K2, D2 = cv_file.getNode("K2").mat(), cv_file.getNode("D2").mat()
R, T = cv_file.getNode("R").mat(), cv_file.getNode("T").mat()
R1, R2 = cv_file.getNode("R1").mat(), cv_file.getNode("R2").mat()
P1, P2 = cv_file.getNode("P1").mat(), cv_file.getNode("P2").mat()
Q = cv_file.getNode("Q").mat()
cv_file.release()

# === Focal length and baseline ===
focal_length_px = P1[0, 0]
baseline_mm = abs(T[0][0])
Fx, Fy = P1[0, 0], P1[1, 1]

# === Load YOLOv8 model ===
model = YOLO('/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/runs/detect/train10/weights/best.pt').to('cuda')

# === Global variables and synchronization ===
bridge = CvBridge()
left_frame = None
right_frame = None
frame_lock = Lock()

# === Stereo Matcher ===
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

# === Callbacks ===
def left_callback(msg):
    global left_frame
    with frame_lock:
        left_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def right_callback(msg):
    global right_frame
    with frame_lock:
        right_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def main():
    global left_frame, right_frame

    rospy.init_node('stereo_depth_yolo_node', anonymous=True)
    rospy.Subscriber("/camera/left/image_raw", Image, left_callback)
    rospy.Subscriber("/camera/right/image_raw", Image, right_callback)
    target_pub = rospy.Publisher("/target_point", Point, queue_size=1)

    rospy.loginfo("Waiting for camera frames...")
    while left_frame is None or right_frame is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    with frame_lock:
        image_size = (left_frame.shape[1], left_frame.shape[0])
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    rate = rospy.Rate(5)
    prev_time = cv2.getTickCount()
    frame_count = 0

    while not rospy.is_shutdown():
        with frame_lock:
            if left_frame is None or right_frame is None:
                continue
            left = left_frame.copy()
            right = right_frame.copy()

        rectL = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

        # Compute disparity every 2nd frame to reduce CPU load
        if frame_count % 2 == 0:
            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
            disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
            points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # Run YOLO on resized image for speed
        detection_input = cv2.resize(rectL, (640, 640))
        results = model(detection_input, verbose=False)[0]

        scale_x = rectL.shape[1] / 640
        scale_y = rectL.shape[0] / 640

        for box in results.boxes:
            if box.conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if 0 <= cx < disparity.shape[1] and 0 <= cy < disparity.shape[0]:
                Z = points_3D[cy, cx, 2]
                if np.isfinite(Z) and Z > 0:
                    label = f"{results.names[int(box.cls[0])]} {Z:.0f}mm"
                    cv2.rectangle(rectL, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rectL, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    w, h = rectL.shape[1], rectL.shape[0]
                    Cx, Cy = w // 2, h // 2
                    X = ((cx - Cx) * Z) / Fx
                    Y = ((cy - Cy) * Z) / Fy
                    target_msg = Point(x=X, y=Y, z=Z)
                    target_pub.publish(target_msg)
                else:
                    rospy.logwarn_throttle(1, f"Invalid depth at object center: ({cx}, {cy})")

        # FPS calculation
        cur_time = cv2.getTickCount()
        time_diff = (cur_time - prev_time) / cv2.getTickFrequency()
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_time = cur_time
        cv2.putText(rectL, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display (disable for headless mode)
        cv2.imshow("YOLO + Depth", rectL)
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("Disparity", disp_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
