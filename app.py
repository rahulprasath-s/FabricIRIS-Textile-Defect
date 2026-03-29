import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- BACKEND: Model Initialization ---
@st.cache_resource
def load_model():
    # Load the CoreML model
    model = YOLO("runs/detect/textile_v1/weights/best.mlpackage")
    return model

model = load_model()

# --- FRONTEND: UI Layout ---
st.set_page_config(page_title="FabricIRIS Inspection", layout="centered")
st.title("🧵 FabricIRIS: Textile Defect Detection")
st.write("Position the fabric in the center and click 'Analyze Fabric'.")

# 1. Camera Input Component
img_file = st.camera_input("Viewfinder")

if img_file:
    # Convert uploaded file to PIL then to OpenCV format
    img = Image.open(img_file)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- MIRROR THE FRAME ---
    # 1 = horizontal flip
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # 2. ROI Logic (Center 20% Crop)
    roi_w, roi_h = int(w * 0.20), int(h * 0.20)
    x1, y1 = (w - roi_w) // 2, (h - roi_h) // 2
    x2, y2 = x1 + roi_w, y1 + roi_h
    
    # 3. Analyze Button
    if st.button("🔍 Analyze Fabric"):
        # Crop to the ROI from the mirrored frame
        roi_crop = frame[y1:y2, x1:x2]
        
        # Run AI Backend
        results = model.predict(roi_crop, conf=0.32, imgsz=640, device="cpu")
        
        # Process Results
        is_defective = False
        for result in results:
            if len(result.boxes) > 0:
                is_defective = True
                break
        
        # 4. Frontend Feedback
        st.divider()
        if is_defective:
            st.error("🚨 RESULT: DEFECT DETECTED")
            # Displaying in RGB for Streamlit's st.image
            st.image(roi_crop, caption="Detected Issue Area", channels="BGR")
        else:
            st.success("✅ RESULT: FABRIC CLEAN")
            st.image(roi_crop, caption="Clean Fabric Area", channels="BGR")

# --- UI Sidebar Info ---
st.sidebar.header("System Status")
st.sidebar.info("Model: YOLO11-CoreML")
st.sidebar.write("Mirror Mode: ENABLED")
st.sidebar.write("Region of Interest: 20% Center")