import cv2
from ultralytics import YOLO

def main():
    # 1. Load Model
    model = YOLO("runs/detect/textile_v1/weights/best.mlpackage")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("\n--- FabricIRIS: Center 20% Focus Mode ---")
    print("1. Place fabric inside the central box.")
    print("2. Press SPACEBAR to Analyze.")
    print("3. Press 'q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 2. CALCULATE 20% CENTER BOX (ROI)
        # We take 20% of the width and height
        roi_w, roi_h = int(w * 0.40), int(h * 0.40)
        x1_roi, y1_roi = (w - roi_w) // 2, (h - roi_h) // 2
        x2_roi, y2_roi = x1_roi + roi_w, y1_roi + roi_h

        # 3. LIVE VIEW WITH GUIDE BOX
        view_img = frame.copy()
        # Draw the target box (White)
        cv2.rectangle(view_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 255, 255), 2)
        cv2.putText(view_img, "PLACE FABRIC HERE", (x1_roi, y1_roi - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("FabricIRIS Viewfinder", view_img)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # 4. CROP THE IMAGE (Only look at the 20% center)
            roi_crop = frame[y1_roi:y2_roi, x1_roi:x2_roi]
            
            # Run AI on the small crop
            results = model.predict(roi_crop, conf=0.5, imgsz=640, device="cpu")
            
            # Check for defects
            is_defective = False
            for result in results:
                if len(result.boxes) > 0:
                    is_defective = True
                    break

            # 5. DISPLAY STATIC RESULT
            result_img = view_img.copy()
            if is_defective:
                msg = "RESULT: DEFECT FOUND"
                color = (0, 0, 255) # Red
                # Highlight the box in Red
                cv2.rectangle(result_img, (x1_roi, y1_roi), (x2_roi, y2_roi), color, 4)
            else:
                msg = "RESULT: FABRIC CLEAN"
                color = (0, 255, 0) # Green
                cv2.rectangle(result_img, (x1_roi, y1_roi), (x2_roi, y2_roi), color, 4)

            # Draw Result Banner
            cv2.rectangle(result_img, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(result_img, msg, (20, 45), cv2.FONT_HERSHEY_TRIPLEX, 1.2, color, 2)

            cv2.imshow("FabricIRIS Viewfinder", result_img)
            print(msg)
            cv2.waitKey(0) # Wait for key press to reset

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()