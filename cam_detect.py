"""
cam_tracker_db.py – FIXED 100%
- No 'can't set attribute id' crash
- Proper filtering without modifying YOLO structures
"""

import os
import time
import cv2
import sqlite3
import numpy as np
from datetime import datetime
from ultralytics import YOLO

DB_PATH = "object_memory.db"
SNAPSHOT_DIR = "snapshots"
MODEL_WEIGHTS = "yolov8n.pt"
CAM_INDEX = 0
CONF_THRESH = 0.45

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# -----------------------------
# DATABASE SETUP
# -----------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_name TEXT,
    tracker_id INTEGER,
    timestamp TEXT,
    bbox TEXT,
    snapshot_path TEXT
)
""")
conn.commit()


def store_detection(object_name, tracker_id, bbox, snapshot):
    ts = datetime.now().isoformat()
    bbox_s = ",".join(map(str, bbox))
    filename = f"{object_name}_{tracker_id}_{int(time.time())}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, snapshot)

    c.execute(
        "INSERT INTO detections (object_name, tracker_id, timestamp, bbox, snapshot_path) VALUES (?, ?, ?, ?, ?)",
        (object_name, tracker_id, ts, bbox_s, path)
    )
    conn.commit()
    return path


# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO(MODEL_WEIGHTS)

# -----------------------------
# CAMERA START
# -----------------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Started Lost Object Finder")
print("Press 'f' to find object")
print("Press 'q' to quit")

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.track(frame, persist=True, verbose=False)
        r = results[0]

        # -----------------------------
        # APPLY CONFIDENCE FILTER SAFELY
        # -----------------------------
        if hasattr(r, "boxes") and r.boxes is not None:

            boxes = r.boxes
            confs = boxes.conf.cpu().numpy()
            keep_idx = np.where(confs >= CONF_THRESH)[0]

            for i in keep_idx:
                box = boxes.xyxy[i].cpu().numpy()
                obj_id = int(boxes.id[i]) if boxes.id is not None else -1
                cls_id = int(boxes.cls[i])
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 200, 16), 2)
                cv2.putText(frame, f"{label} ID:{obj_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 200, 16), 2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_small = cv2.resize(crop, (224, 224))
                snap = store_detection(label, obj_id, (x1, y1, x2, y2), crop_small)

        cv2.imshow("Lost Object Finder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('f'):
            name = input("Find object >>> ").strip().lower()
            if not name:
                continue

            c.execute(
                "SELECT object_name, tracker_id, timestamp, bbox, snapshot_path "
                "FROM detections WHERE object_name LIKE ? ORDER BY id DESC LIMIT 1",
                (f"%{name}%",)
            )
            row = c.fetchone()

            if not row:
                print("No record found.")
            else:
                obj_name, tracker_id, ts, bbox_s, snap = row
                x1, y1, x2, y2 = map(int, bbox_s.split(","))

                snap_img = cv2.imread(snap)
                cv2.imshow("Last Seen Snapshot", snap_img)

                highlight = frame.copy()
                cv2.rectangle(highlight, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.imshow("Lost Object Finder", highlight)
                cv2.waitKey(0)

finally:
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    print("Closed cleanly.")
