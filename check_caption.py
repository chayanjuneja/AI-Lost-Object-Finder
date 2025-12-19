# check_caption.py
from captioner import caption_image_path
import os, sys

snap_dir = "snapshots"
if not os.path.isdir(snap_dir):
    print("No snapshots folder found.")
    sys.exit(0)

files = sorted(os.listdir(snap_dir))
if not files:
    print("No snapshots yet, skip.")
    sys.exit(0)

path = os.path.join(snap_dir, files[0])
print("Testing on:", files[0])
print("Caption:", caption_image_path(path))
