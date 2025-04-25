#!/usr/bin/env python
import rospy
import cv2
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge

class DisparityViewer:
    def __init__(self):
        rospy.init_node("disparity_viewer", anonymous=True)
        self.bridge = CvBridge()
        rospy.Subscriber("/stereo/disparity", DisparityImage, self.disparity_callback)

    def disparity_callback(self, msg):
        try:
            # Extract actual disparity image from message
            disp_image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")

            # Normalize for visualization
            disp_normalized = cv2.normalize(disp_image, None, 0, 255, cv2.NORM_MINMAX)
            disp_normalized = disp_normalized.astype('uint8')

            # Show image
            cv2.imshow("Disparity (Depth)", disp_normalized)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Error converting disparity image: %s", e)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = DisparityViewer()
    viewer.run()
