from pathlib import Path

import cv2
from ultralytics import YOLO

BOX_COLORS = {
    0: (0, 140, 255),
    1: (0, 0, 255),
    2: (0, 215, 255),
}

CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 640
CENTER_ROI_RATIO = 0.30


def resolve_model_path():
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "runs" / "detect" / "textile_v1" / "weights" / "best.mlpackage",
        base_dir / "runs" / "detect" / "textile_v1" / "weights" / "best.pt",
        base_dir.parent / "runs" / "detect" / "textile_v1" / "weights" / "best.mlpackage",
        base_dir.parent / "runs" / "detect" / "textile_v1" / "weights" / "best.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Model not found. Expected best.mlpackage or best.pt under "
        "'runs/detect/textile_v1/weights/'."
    )


def get_center_roi(frame):
    frame_h, frame_w = frame.shape[:2]
    roi_w = max(1, int(frame_w * CENTER_ROI_RATIO))
    roi_h = max(1, int(frame_h * CENTER_ROI_RATIO))
    x1 = (frame_w - roi_w) // 2
    y1 = (frame_h - roi_h) // 2
    return x1, y1, x1 + roi_w, y1 + roi_h


def draw_label(frame, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness = 1
    text_padding = 6
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    frame_h, frame_w = frame.shape[:2]

    label_x1 = max(x1, 0)
    label_y1 = y1 - text_size[1] - (text_padding * 2)
    if label_y1 < 0:
        label_y1 = min(y1 + 8, frame_h - text_size[1] - (text_padding * 2) - baseline)

    label_y2 = label_y1 + text_size[1] + (text_padding * 2) + baseline
    label_x2 = min(label_x1 + text_size[0] + (text_padding * 2), frame_w - 1)
    label_y2 = min(label_y2, frame_h - 1)

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        frame,
        text,
        (label_x1 + text_padding, label_y2 - text_padding - baseline),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )


def draw_detections(frame, results, offset_x=0, offset_y=0):
    defect_count = 0
    frame_h, frame_w = frame.shape[:2]

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y

            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(box.cls[0]) if box.cls is not None else -1
            color = BOX_COLORS.get(cls_id, (0, 0, 255))
            label_text = f"DEFECT {confidence:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label_text, x1, max(y1 - 8, 0), color)
            defect_count += 1

    return defect_count


def main():
    try:
        model_path = resolve_model_path()
    except FileNotFoundError as error:
        print(error)
        return

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("FabricIRIS Live: detecting defects in real time. Press 'q' to quit.")
    print(f"Using model: {model_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            break

        frame = cv2.flip(frame, 1)
        roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(frame)
        roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        results = model.predict(
            roi_crop,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )

        defect_count = draw_detections(
            frame,
            results,
            offset_x=roi_x1,
            offset_y=roi_y1,
        )

        guide_color = (0, 0, 255) if defect_count else (255, 255, 255)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), guide_color, 2)
        cv2.putText(
            frame,
            "CENTER 30% DETECTION ZONE",
            (roi_x1, max(roi_y1 - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            guide_color,
            2,
        )

        status_color = (0, 0, 255) if defect_count else (0, 180, 0)
        status_text = "DEFECT DETECTED" if defect_count else "NO DEFECT DETECTED"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (20, 20, 20), -1)
        cv2.putText(
            frame,
            status_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
        )
        cv2.putText(
            frame,
            f"DETECTION BOXES: {defect_count}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("FabricIRIS - Textile Inspection", frame)

        if defect_count > 0:
            print("Defect detected! Video paused. Press 'q' to quit or any other key to resume.")
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
