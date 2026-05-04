import cv2
import mediapipe as mp
import math
import time
import os

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)

hands = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def set_volume(vol):
    vol = max(0, min(100, int(vol)))
    os.system(f"amixer set Master {vol}% > /dev/null 2>&1")

while True:
    success, img = cap.read()
    if not success:
        break

    timestamp = int(time.time() * 1000)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img
    )

    result = hands.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h, w, _ = img.shape

            # Get thumb tip and index tip
            thumb = hand[4]
            index = hand[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Draw points
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Distance
            length = math.hypot(x2 - x1, y2 - y1)

            # Map distance → volume (tune these values)
            vol = (length - 20) * (100 / 200)

            set_volume(vol)

            # Show volume
            cv2.putText(img, f'Vol: {int(vol)}%', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()