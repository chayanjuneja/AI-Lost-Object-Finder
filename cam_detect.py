"""
cam_tracker_db.py – FIXED 100%
- No 'can't set attribute id' crash
- Proper filtering without modifying YOLO structures
"""
import time
from captioner import caption_image_path
from utils import get_zone_from_bbox
import db
import os
import time
import cv2
import sqlite3
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO

_caption_ts = {}       # tracker_id -> last caption timestamp
CAPTION_COOLDOWN = 30  # seconds between captions per tracker (adjust if CPU slow)

DB_PATH = "object_memory.db"
SNAPSHOT_DIR = "snapshots"
MODEL_WEIGHTS = "yolov8n.pt"
CAM_INDEX = 0
CONF_THRESH = 0.45

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# -----------------------------
# DATABASE SETUP
# -----------------------------
import threading
import queue

# Queue for paths that need captioning: (snapshot_path, detection_row_id)
caption_queue = queue.Queue()

# Thread-safe store for completed captions: snapshot_path -> (caption, detection_row_id)
caption_results = {}

def in_window_text_input_nonblocking(window_name, cap, prompt="Find object >>> "):
    text = ""
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, prompt + text, (10, h-10), font, 0.7, (255,255,255), 2)

        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1) & 0xFF

        if k in [13, 10]:   # Enter
            return text.strip()

        if k == 27:         # ESC
            return None

        if k in [8, 127]:   # Backspace
            text = text[:-1]
            continue

        if 32 <= k <= 126:  # printable characters
            text += chr(k)
            continue

def caption_worker():
    while True:
        item = caption_queue.get()
        if item is None:
            # poison pill — exit thread
            caption_queue.task_done()
            break
        snapshot_path, detection_row_id = item
        try:
            cap_text = caption_image_path(snapshot_path)
        except Exception:
            cap_text = ""
        # store result
        caption_results[snapshot_path] = (cap_text, detection_row_id)
        caption_queue.task_done()

# start worker thread (daemon so it won't block exit)
caption_thread = threading.Thread(target=caption_worker, daemon=True)
caption_thread.start()


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
                snapshot_path = db.store_image_snapshot(crop_small, path_hint=f"{label}_{obj_id}")
                zone = get_zone_from_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])

                # Insert detection WITHOUT caption (will be filled later)
                record_id = db.insert_detection(label, obj_id, (x1, y1, x2, y2), snapshot_path, caption=None, zone=zone)

                # Queue the snapshot for background captioning
                caption_queue.put((snapshot_path, record_id))

                # update named object last_locations (caption will be updated later)
                obj_named_id = db.update_object(label, label)
                db.add_movement(obj_named_id, x1, y1, x2=x2, y2=y2, snapshot_path=snapshot_path, caption=None, zone=zone, tracker_id=obj_id)


        if caption_results:
            # copy keys to avoid mutation during iteration
            ready_keys = list(caption_results.keys())
            for snap_path in ready_keys:
                cap_text, rec_id = caption_results.pop(snap_path)
                try:
                    with db.get_db() as conn:
                        cur = conn.cursor()
                        cur.execute("UPDATE detections SET caption=? WHERE id=?", (cap_text, rec_id))
                        # also update last_locations snapshot/caption for named_objects if needed:
                        # find named_object id(s) that reference this snapshot path and update last_locations entries
                        cur.execute("SELECT id FROM named_objects WHERE snapshot_path = ? LIMIT 1", (snap_path,))
                        row = cur.fetchone()
                        if row:
                            # simple update — prepend to last_locations json using Python then save
                            obj_id = row["id"]
                            cur.execute("SELECT last_locations FROM named_objects WHERE id = ? LIMIT 1", (obj_id,))
                            lr = cur.fetchone()["last_locations"]
                            try:
                                arr = json.loads(lr) if lr else []
                            except Exception:
                                arr = []
                            # update any matching entries with same snapshot_path
                            for loc in arr:
                                if loc.get("snapshot_path") == snap_path:
                                    loc["caption"] = cap_text
                            cur.execute("UPDATE named_objects SET last_locations = ? WHERE id = ?", (json.dumps(arr), obj_id))
                except Exception:
                    # if DB update fails for any reason, ignore and continue
                    pass
        cv2.imshow("Lost Object Finder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('f'):
            name = in_window_text_input_nonblocking("Lost Object Finder", cap)
            if not name:
                continue

            name = name.lower()

            row = db.get_last_detection_of_label(name)
            if not row:
                print("No record found.")
            else:
                x1, y1, x2, y2 = row["bbox"]
                snap = row["snapshot_path"]

                snap_img = cv2.imread(snap)
                if snap_img is not None:
                    cv2.imshow("Last Seen Snapshot", snap_img)

                highlight = frame.copy()
                cv2.rectangle(highlight, (x1, y1), (x2, y2), (0,0,255), 3)
                cv2.imshow("Lost Object Finder", highlight)
                cv2.waitKey(0)
            if not name:
                continue

            row = db.get_last_detection_of_label(name)
            if not row:
                print("No record found.")
            else:
                obj_name = row["object_label"]
                tracker_id = row["tracker_id"]
                ts = row["timestamp"]
                x1, y1, x2, y2 = row["bbox"]
                snap = row["snapshot_path"]

                snap_img = cv2.imread(snap) if snap else None
                if snap_img is not None:
                    cv2.imshow("Last Seen Snapshot", snap_img)

                highlight = frame.copy()
                cv2.rectangle(highlight, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.imshow("Lost Object Finder", highlight)
                cv2.waitKey(0)

finally:
    cap.release()
    cv2.destroyAllWindows()
    # stop caption thread gracefully
    caption_queue.put(None)
    caption_thread.join(timeout=2)
    print("Closed cleanly.")
