#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class StereoDepthNode:
    def __init__(self):
        rospy.init_node('stereo_depth_node')

        # Load stereo calibration parameters
        cv_file = cv2.FileStorage("/home/shanoj/catkin_ws/src/turret_gun_description/script/stereo_params.xml", cv2.FILE_STORAGE_READ)
        self.K1 = cv_file.getNode("K1").mat()
        self.D1 = cv_file.getNode("D1").mat()
        self.K2 = cv_file.getNode("K2").mat()
        self.D2 = cv_file.getNode("D2").mat()
        self.R = cv_file.getNode("R").mat()
        self.T = cv_file.getNode("T").mat()
        self.R1 = cv_file.getNode("R1").mat()
        self.R2 = cv_file.getNode("R2").mat()
        self.P1 = cv_file.getNode("P1").mat()
        self.P2 = cv_file.getNode("P2").mat()
        self.Q = cv_file.getNode("Q").mat()
        cv_file.release()

        self.bridge = CvBridge()
        self.image_size = None
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None

        self.left_image = None
        self.right_image = None

        self.stereo = cv2.StereoSGBM_create(
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

        rospy.Subscriber("/camera/left/image_raw", Image, self.left_callback)
        rospy.Subscriber("/camera/right/image_raw", Image, self.right_callback)

        self.disp_pub = rospy.Publisher("/stereo/disparity", Image, queue_size=10)

        rospy.loginfo("Stereo depth node initialized.")
        rospy.spin()

    def left_callback(self, msg):
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.try_process()
        except Exception as e:
            rospy.logerr(f"Left image conversion failed: {e}")

    def right_callback(self, msg):
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.try_process()
        except Exception as e:
            rospy.logerr(f"Right image conversion failed: {e}")

    def try_process(self):
        if self.left_image is None or self.right_image is None:
            return

        if self.image_size is None:
            h, w = self.left_image.shape[:2]
            self.image_size = (w, h)
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.K1, self.D1, self.R1, self.P1, self.image_size, cv2.CV_32FC1)
            self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.K2, self.D2, self.R2, self.P2, self.image_size, cv2.CV_32FC1)

        # Rectify
        rectL = cv2.remap(self.left_image, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(self.right_image, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # Grayscale
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        # Disparity
        disparity = self.stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)

        # # Center pixel depth
        h, w = disparity.shape
        print("%.2f, %.2f",(h,w))
        center_disp = disparity[h // 2, w // 2]
        # print(center_disp)
        if center_disp > 0:
            Z = points_3D[h // 2, w // 2, 2]
            rospy.loginfo(f"Center disparity: {center_disp:.2f}, Depth (Z): {Z:.2f} mm")
        else:
            rospy.loginfo("Invalid disparity at center pixel.")

        # Visualization
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        cv2.circle(disp_vis, (w // 2, h // 2), 5, (255, 0, 0), 2)

        # Publish disparity image
        try:
            disp_msg = self.bridge.cv2_to_imgmsg(disp_vis, encoding="mono8")
            self.disp_pub.publish(disp_msg)
        except Exception as e:
            rospy.logerr(f"Disparity publish failed: {e}")

if __name__ == '__main__':
    try:
        StereoDepthNode()
    except rospy.ROSInterruptException:
        pass
