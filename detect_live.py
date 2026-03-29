import cv2
from ultralytics import YOLO

def main():
    # 1. Load Model
    model = YOLO("runs/detect/textile_v1/weights/best.mlpackage")
    
    # 2. Local Name Mapping (Safest way for CoreML)
    my_labels = {0: "OIL SPOT", 1: "HOLE", 2: "TORN SPOT"}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("FabricIRIS Live: Printing Class Names. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Mirror view

        # 3. Run Inference (Using CPU for stability if MPS was failing)
        results = model.track(frame, persist=True, conf=0.15, imgsz=640, device="cpu")

        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                # Get Box Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get the Class Name from our local dictionary
                cls_id = int(box.cls[0])
                label_text = my_labels.get(cls_id, "UNKNOWN")
                confidence = float(box.conf[0])

                # --- PRINTING LOGIC ---
                
                # A. Draw the Bounding Box (Keep it thin/grey so text stands out)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # B. Create the string: "HOLE 85%"
                full_label = f"{label_text} {confidence:.0%}"

                # C. Draw a Background Label Tag (Top-left of the box)
                # This makes the text readable regardless of the fabric color
                cv2.rectangle(frame, (x1, y1 - 35), (x1 + 220, y1), (0, 0, 0), -1) 
                
                # D. Write the actual Class Name
                cv2.putText(
                    frame, 
                    full_label, 
                    (x1 + 5, y1 - 10),            # Position
                    cv2.FONT_HERSHEY_DUPLEX,      # Font style
                    0.8,                          # Font scale
                    (255, 255, 255),              # White text color
                    2                             # Thickness
                )

        # UI Overlay
        cv2.putText(frame, "ANALYZING FABRIC...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("FabricIRIS - Textile Inspection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()