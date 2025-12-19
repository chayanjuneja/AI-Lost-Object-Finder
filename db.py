# db.py
"""
Step-2 DB module with caption & zone support.

Tables:
- detections: stores each detection (object_label, tracker_id, timestamp, bbox, snapshot_path, caption, zone)
- named_objects: stores named objects and last_locations (JSON array, most recent first) with caption & zone

Functions:
- init_db()
- store_image_snapshot(img, path_hint)
- insert_detection(object_label, tracker_id, bbox, snapshot_path, caption=None, zone=None)
- update_named_object(...)
- get_named_object(name)
- get_last_detection_of_label(label)
- update_object(name, object_label)
- add_movement(obj_id, x1, y1, x2=None, y2=None, snapshot_path=None, caption=None, zone=None, tracker_id=None)
"""
import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

DB_PATH = "object_memory.db"
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
_MAX_LOCATIONS = 3

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db(path: str = DB_PATH):
    conn = _get_connection(path)
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()

def _push_location_json(existing_json: Optional[str], loc_entry: Dict[str, Any], max_len: int = _MAX_LOCATIONS) -> str:
    arr = []
    if existing_json:
        try:
            arr = json.loads(existing_json)
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
    arr.insert(0, loc_entry)
    arr = arr[:max_len]
    return json.dumps(arr)

def init_db(db_path: str = DB_PATH) -> None:
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_label TEXT NOT NULL,
                tracker_id INTEGER,
                timestamp TEXT NOT NULL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                snapshot_path TEXT,
                caption TEXT,
                zone TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS named_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                object_label TEXT NOT NULL,
                tracker_id INTEGER,
                last_seen_ts TEXT,
                last_x1 INTEGER,
                last_y1 INTEGER,
                last_x2 INTEGER,
                last_y2 INTEGER,
                snapshot_path TEXT,
                last_locations TEXT
            );
            """
        )

# ensure DB
try:
    init_db()
except Exception:
    pass

def store_image_snapshot(img: np.ndarray, path_hint: str = "snap", quality: int = 85) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = int(datetime.utcnow().timestamp())
    fname = f"{path_hint}_{ts}.jpg"
    path = os.path.join(SNAPSHOT_DIR, fname)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return path

def insert_detection(object_label: str, tracker_id: Optional[int], bbox: Tuple[int,int,int,int],
                     snapshot_path: Optional[str], caption: Optional[str] = None, zone: Optional[str] = None,
                     camera_id: Optional[str] = None, db_path: str = DB_PATH) -> int:
    x1, y1, x2, y2 = map(int, bbox)
    ts = _now_iso()
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO detections (object_label, tracker_id, timestamp, x1, y1, x2, y2, snapshot_path, caption, zone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (object_label, tracker_id, ts, x1, y1, x2, y2, snapshot_path, caption, zone),
        )
        return cur.lastrowid

def update_named_object(name: str, object_label: str, tracker_id: Optional[int],
                        bbox: Tuple[int,int,int,int], snapshot_path: Optional[str],
                        caption: Optional[str] = None, zone: Optional[str] = None,
                        camera_id: Optional[str] = None, db_path: str = DB_PATH) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    ts = _now_iso()
    loc_entry = {
        "timestamp": ts,
        "bbox": [x1, y1, x2, y2],
        "snapshot_path": snapshot_path,
        "caption": caption,
        "zone": zone,
        "tracker_id": tracker_id,
        "camera_id": camera_id
    }
    last_locations_json = json.dumps([loc_entry])
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE named_objects
            SET object_label = ?, tracker_id = ?, last_seen_ts = ?, last_x1 = ?, last_y1 = ?, last_x2 = ?, last_y2 = ?, snapshot_path = ?, last_locations = ?
            WHERE name = ?
            """,
            (object_label, tracker_id, ts, x1, y1, x2, y2, snapshot_path, last_locations_json, name)
        )
        if cur.rowcount == 0:
            cur.execute(
                """
                INSERT INTO named_objects (name, object_label, tracker_id, last_seen_ts, last_x1, last_y1, last_x2, last_y2, snapshot_path, last_locations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, object_label, tracker_id, ts, x1, y1, x2, y2, snapshot_path, last_locations_json)
            )

def get_named_object(name: str, db_path: str = DB_PATH) -> Optional[Dict[str,Any]]:
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM named_objects WHERE name = ? LIMIT 1", (name,))
        row = cur.fetchone()
        if not row:
            return None
        last_locations = []
        if row["last_locations"]:
            try:
                last_locations = json.loads(row["last_locations"])
            except Exception:
                last_locations = []
        return {
            "id": row["id"],
            "name": row["name"],
            "object_label": row["object_label"],
            "tracker_id": row["tracker_id"],
            "last_seen_ts": row["last_seen_ts"],
            "bbox": (row["last_x1"], row["last_y1"], row["last_x2"], row["last_y2"]),
            "snapshot_path": row["snapshot_path"],
            "last_locations": last_locations
        }

def get_last_detection_of_label(label: str, db_path: str = DB_PATH) -> Optional[Dict[str,Any]]:
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM detections
            WHERE object_label = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (label,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "object_label": row["object_label"],
            "tracker_id": row["tracker_id"],
            "timestamp": row["timestamp"],
            "bbox": (row["x1"], row["y1"], row["x2"], row["y2"]),
            "snapshot_path": row["snapshot_path"],
            "caption": row["caption"],
            "zone": row["zone"]
        }

def update_object(name: str, object_label: str) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM named_objects WHERE name = ? LIMIT 1", (name,))
        row = cur.fetchone()
        if row:
            return int(row["id"])
        cur.execute(
            """
            INSERT INTO named_objects (name, object_label)
            VALUES (?, ?)
            """,
            (name, object_label)
        )
        return cur.lastrowid

def add_movement(obj_id: int, x1: int, y1: int, x2: Optional[int] = None, y2: Optional[int] = None,
                 snapshot_path: Optional[str] = None, caption: Optional[str] = None, zone: Optional[str] = None,
                 tracker_id: Optional[int] = None, camera_id: Optional[str] = None, db_path: str = DB_PATH) -> None:
    if x2 is None: x2 = x1
    if y2 is None: y2 = y1
    ts = _now_iso()
    loc_entry = {
        "timestamp": ts,
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "snapshot_path": snapshot_path,
        "caption": caption,
        "zone": zone,
        "tracker_id": tracker_id,
        "camera_id": camera_id
    }
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name, last_locations FROM named_objects WHERE id = ? LIMIT 1", (obj_id,))
        row = cur.fetchone()
        if not row:
            return
        name = row["name"]
        existing = row["last_locations"]
        new_json = _push_location_json(existing, loc_entry, max_len=_MAX_LOCATIONS)
        cur.execute(
            """
            UPDATE named_objects
            SET tracker_id = ?, last_seen_ts = ?, last_x1 = ?, last_y1 = ?, last_x2 = ?, last_y2 = ?, snapshot_path = ?, last_locations = ?
            WHERE id = ?
            """,
            (tracker_id, ts, int(x1), int(y1), int(x2), int(y2), snapshot_path, new_json, obj_id)
        )

def list_named_objects(db_path: str = DB_PATH) -> List[Dict[str,Any]]:
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, object_label, last_seen_ts FROM named_objects ORDER BY last_seen_ts DESC")
        return [dict(r) for r in cur.fetchall()]

def delete_named_object(name: str, db_path: str = DB_PATH) -> bool:
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM named_objects WHERE name = ?", (name,))
        return cur.rowcount > 0
