import cv2
import torch
from torchvision import models, transforms
import numpy as np

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Faster R-CNN model pre-trained on COCO dataset
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define COCO classes (91 classes with 'N/A' for unused)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Set up video capture
video = cv2.VideoCapture(0)  # Use 0 for the default camera

# Define a transformation to preprocess the frames
transform = transforms.Compose([
    transforms.ToTensor()
])

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to RGB (from BGR) and apply the transformation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb_frame).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform object detection
    with torch.no_grad():
        predictions = model(input_tensor)[0]  # Get the predictions for the single frame

    # Filter out predictions with a low score
    threshold = 0.8  # Confidence threshold
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= threshold:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Get the label name
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            cv2.putText(frame, f"{label_name} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the video capture and close windows
video.release()
cv2.destroyAllWindows()
