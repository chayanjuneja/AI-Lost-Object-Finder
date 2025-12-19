# key_test.py
import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit(1)

print("Click the camera window to focus it, then press keys. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("KEY TEST", frame)
    k = cv2.waitKey(1) & 0xFF
    if k != 255:
        print("key code:", k)
    if k == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
