import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/runs/detect/train10/weights/best.pt')  # Provide the path to your 'best.pt' model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera, change if you have other cameras connected

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform object detection
    results = model(frame)

    # Render the results and get the annotated image
    annotated_frame = results[0].plot()  # This is where the error was
    # 'results[0]' is the first (and usually the only) result, and '.plot()' is the correct method

    # Display the frame with detections
    cv2.imshow("Human Detection", annotated_frame)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
