import sqlite3

conn = sqlite3.connect('object_memory.db')
rows = list(conn.execute("SELECT id, object_label, tracker_id, timestamp, snapshot_path, caption, zone FROM detections ORDER BY id DESC LIMIT 10"))
conn.close()

for r in rows:
    print(r)
