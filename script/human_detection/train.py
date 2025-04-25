from ultralytics import YOLO

# Load the pre-trained YOLO model (smallest version for testing, 'yolov8n.pt')
model = YOLO('yolov8n.pt')  # Load pretrained weights instead of the model architecture YAML

# Train the model on the GPU (if available) with the correct dataset configuration
results = model.train(
    data='/home/shanoj/catkin_ws/src/turret_gun_description/script/human_detection/data.yaml',  # Path to your dataset YAML file
    epochs=100,  # Set number of epochs (adjust based on your needs)
    imgsz=416,  # Image size for training (you can adjust if needed)
    device='cuda',  # Use GPU if available, otherwise it will use CPU
    batch=4,  # Adjust the batch size based on your hardware and dataset
    cache=True,  # Cache images for faster training on the next run
)

# Optionally print out results after training
print(results)  # Displays the results from training
