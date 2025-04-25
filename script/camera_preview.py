#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraObject:
    def __init__(self):
        rospy.init_node("object_detection", anonymous=False)
        self.bridge = CvBridge()
        rospy.Subscriber('/left_camera1/image_raw', Image, self.image_callback)

        # Load YOLO model for object detection
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Get output layer names for YOLO
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.classes = []  # List of class names
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.bbox = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(cv_image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Forward pass to get bounding boxes, confidences, and class IDs
        detections = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        # Process YOLO detections
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * cv_image.shape[1])
                    center_y = int(detection[1] * cv_image.shape[0])
                    w = int(detection[2] * cv_image.shape[1])
                    h = int(detection[3] * cv_image.shape[0])

                    # Bounding box coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        self.bbox = []
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            self.bbox.append([x, y, x + w, y + h])
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(cv_image, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("object_detection", cv_image)
        cv2.waitKey(1)

    def track_object(self):
        if self.bbox and len(self.bbox[0]) > 3:
            center_x = (self.bbox[0][0] + self.bbox[0][2]) / 2
            center_y = (self.bbox[0][1] + self.bbox[0][3]) / 2
            print("Object center:", center_x, center_y)

if __name__ == '__main__':
    try:
        camera_handler = CameraObject()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if camera_handler.bbox is not None:
                camera_handler.track_object()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        pass
