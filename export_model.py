from ultralytics import YOLO

# Load your trained custom model
model = YOLO("runs/detect/textile_v1/weights/best.pt")

# Export to CoreML format
# nms=True helps the model handle overlapping boxes internally
model.export(format="coreml", nms=True, imgsz=[640,640])