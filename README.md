# FabricIRIS: Textile Defect Detection

An Edge-AI solution running on Apple Silicon (M1) to detect manufacturing defects in real-time.

## Features
- Detects Holes, Knots, Lines, and Stains.
- Optimized for **Apple Neural Engine** using Core ML.
- High-speed inference on MacBook Air M1.

## Setup
1. Install dependencies: `pip install ultralytics coremltools opencv-python`
2. Train the model: `python train.py`
3. Export to Core ML: `python export_model.py`
4. Run live: `python detect_live.py`
5. Snap & Analyse: `python detect_snap.py`
