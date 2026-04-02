from ultralytics import YOLO

# Load the base YOLO11 model (it will auto-download the .pt file once)
model = YOLO("yolo11n.pt") 

# Train locally using your downloaded dataset
model.train(
    data=f"{dataset.location}/data.yaml", 
    epochs=50, 
    imgsz=640, 
    device=0  # Use 'cpu' if you don't have an NVIDIA GPU
)