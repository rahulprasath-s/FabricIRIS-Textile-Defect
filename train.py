from ultralytics import YOLO
import os

# This environment variable helps with some MPS memory issues
os.environ ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def start_training():
    dataset_config = "/Users/rahul/FabricIRIS/textile_defect_detection/data.yaml"
    
    # Load the Nano model
    model = YOLO("yolo11n.pt") 

    model.train(
        data=dataset_config,
        epochs=50,
        imgsz=640,
        batch=4,          # REDUCED: Smaller batch is more stable for M1 memory
        device="cpu",
        name="textile_v1",
        # NEW SETTINGS FOR STABILITY:
        workers=0,        # Prevents multi-threading memory conflicts on Mac
        exist_ok=True,    # Overwrites the folder if it crashed mid-way
        amp=False         # Disables Automatic Mixed Precision (often causes MPS errors)
    )

if __name__ == "__main__":
    start_training()