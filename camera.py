import cv2
from ultralytics import YOLO
from db import update_object, add_movement
from utils import get_zone_from_bbox
import os

SNAP_DIR = "snapshots/"
os.makedirs(SNAP_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

def start_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found.")
        return

    print("Camera started... (Press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        results = model.track(frame, persist=True, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                label = model.names[cls]

                zone = get_zone_from_bbox(x1, y1, x2, y2, frame_w, frame_h)

                obj_id = update_object(label, label)
                add_movement(obj_id, x1, y1)

                cv2.putText(frame, f"{label} (ID:{track_id})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("Lost Object Finder", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
