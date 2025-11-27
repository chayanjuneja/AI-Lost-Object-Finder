import sqlite3
from datetime import datetime

DB_NAME = "objects.db"

def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            object_name TEXT,
            last_seen_time TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS movement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id INTEGER,
            x INTEGER,
            y INTEGER,
            time TEXT,
            FOREIGN KEY (object_id) REFERENCES objects(id)
        )
    """)

    conn.commit()
    conn.close()

def update_object(label, object_name):
    now = datetime.now().isoformat()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM objects WHERE label=?", (label,))
    row = cur.fetchone()

    if row:
        obj_id = row[0]
        cur.execute("UPDATE objects SET last_seen_time=? WHERE id=?", (now, obj_id))
    else:
        cur.execute(
            "INSERT INTO objects(label, object_name, last_seen_time) VALUES (?, ?, ?)",
            (label, object_name, now)
        )
        obj_id = cur.lastrowid

    conn.commit()
    conn.close()
    return obj_id

def add_movement(object_id, x, y):
    now = datetime.now().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO movement(object_id, x, y, time) VALUES (?,?,?,?)",
        (object_id, x, y, now)
    )
    conn.commit()
    conn.close()
