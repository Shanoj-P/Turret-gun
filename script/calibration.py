#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

class StereoCalibrator:
    def __init__(self):
        # Chessboard configuration
        self.chessboard_size = (9, 6)
        self.square_size = 0.025

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((self.chessboard_size[0]*self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []

        self.bridge = CvBridge()
        self.image_count = 0
        self.max_images = 20

        # Subscribers
        left_sub = Subscriber('/stereo/left/image_raw', Image)
        right_sub = Subscriber('/stereo/right/image_raw', Image)
        self.ts = ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        rospy.loginfo("StereoCalibrator initialized. Waiting for synchronized image pairs...")

    def image_callback(self, left_msg, right_msg):
        if self.image_count >= self.max_images:
            return

        try:
            cv_left = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
            cv_right = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        except Exception as e:
            rospy.logerr("CvBridge Error: %s", e)
            return

        gray_left = cv2.cvtColor(cv_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(cv_right, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, None)

        if ret_left and ret_right:
            self.objpoints.append(self.objp)

            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)

            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            self.image_count += 1

            rospy.loginfo(f"Captured {self.image_count} image pairs")

        if self.image_count == self.max_images:
            self.calibrate(gray_left.shape[::-1])

    def calibrate(self, image_size):
        rospy.loginfo("Calibrating cameras...")

        ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_size, None, None)
        ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_size, None, None)

        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            mtx_l, dist_l,
            mtx_r, dist_r,
            image_size,
            criteria=self.criteria,
            flags=flags
        )

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_l, dist_l,
            mtx_r, dist_r,
            image_size,
            R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        fs = cv2.FileStorage("stereo_calibration_ros.xml", cv2.FILE_STORAGE_WRITE)
        fs.write("K1", mtx_l)
        fs.write("D1", dist_l)
        fs.write("K2", mtx_r)
        fs.write("D2", dist_r)
        fs.write("R", R)
        fs.write("T", T)
        fs.write("E", E)
        fs.write("F", F)
        fs.write("R1", R1)
        fs.write("R2", R2)
        fs.write("P1", P1)
        fs.write("P2", P2)
        fs.write("Q", Q)
        fs.release()

        rospy.loginfo("Calibration complete. Results saved to stereo_calibration_ros.xml")


if __name__ == '__main__':
    rospy.init_node('stereo_calibrator', anonymous=True)
    calibrator = StereoCalibrator()
    rospy.spin()
