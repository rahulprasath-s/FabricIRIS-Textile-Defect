from ultralytics import YOLO

# 1. Load your custom trained weights
model = YOLO("custom_yolo11.pt")

# 2. Run inference
results = model.predict(source="0", show=True)