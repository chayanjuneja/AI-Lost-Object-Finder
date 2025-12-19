# check_object.py
import sqlite3, os, sys
from datetime import datetime

DB='object_memory.db'
label = sys.argv[1] if len(sys.argv)>1 else 'book'

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
row = conn.execute(
    "SELECT timestamp, x1,y1,x2,y2, snapshot_path, zone, caption FROM detections WHERE object_label = ? ORDER BY timestamp DESC LIMIT 1",
    (label,)
).fetchone()
conn.close()

if not row:
    print(f"No record found for '{label}'.")
    sys.exit(0)

ts = row['timestamp']
zone = row['zone'] or 'unknown'
snap = row['snapshot_path'] or ''
caption = row['caption'] or ''
print(f"Last seen: '{label}' at {ts} — zone: {zone}")
if caption:
    print("Caption:", caption)
if snap:
    print("Snapshot:", snap)
    # On Windows you can open the snapshot with the default image viewer:
    print("To open the snapshot run (PowerShell):")
    print(f"Start-Process \"{snap}\"")
